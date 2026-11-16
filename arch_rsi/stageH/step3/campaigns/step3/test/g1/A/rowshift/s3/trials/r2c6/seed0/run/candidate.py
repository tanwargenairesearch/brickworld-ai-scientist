import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 48, "n_minibatches": 24, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

FD = 16
STEM = 48
SC = 32
W = 40
PROJ = 32
HID = 32
G = 64
RG = 1.4142135623730951


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def norm(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        n = self.Y * self.X
        fin = cin * n
        self.H = max(32, min(128, 400000 // fin))
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.flat1 = init_(nn.Linear(fin, self.H), RG)
        self.fln = nn.LayerNorm(self.H)
        self.flat2 = init_(nn.Linear(self.H, FD * n), 1.0)
        self.stem1 = init_(nn.Conv2d(cin, STEM, 1), RG)
        self.stem2 = init_(nn.Conv2d(STEM, SC, 1), RG)
        self.c1 = init_(nn.Conv2d(SC + FD, W, 3, padding=1), RG)
        self.n1 = norm(W)
        self.c2 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = norm(W)
        self.proj = init_(nn.Conv2d(W, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        gin = 2 * W + self.H
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.dense = init_(nn.Linear(self.H + G, int(spec.n_place)), 0.01)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        hf = F.relu(self.fln(self.flat1(obs.flatten(1))))
        fm = self.flat2(hf).view(B, FD, Y, X)
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(torch.cat([s, fm], dim=1))))
        f = F.relu(x + self.n2(self.c2(x)))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), hf], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat(fp_pool(p, dy, dx), dim=1)
            z = F.relu(self.mix[i](z))
            res.append(self.outs[i](z) + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1) + self.dense(torch.cat([hf, g], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
