import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 1024, "rollout_len": 8, "n_minibatches": 32, "ppo_epochs": 3, "ent_coef": 0.01, "gamma": 0.9, "gae_lambda": 0.8, "clip_eps": 0.2}

M = 48
W = 40
RAW = 16
P = 16
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


class FpConv(nn.Module):
    def __init__(self, cin, cout, dy, dx, gain):
        super().__init__()
        self.dy = int(dy)
        self.dx = int(dx)
        self.conv = init_(nn.Conv2d(cin, cout, (self.dy + 2, self.dx + 2)), gain)

    def forward(self, p):
        return self.conv(F.pad(p, (1, self.dx, 1, self.dy)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.sa = init_(nn.Conv2d(cin, M, 1), 1.0)
        self.sb = init_(nn.Conv2d(cin, M, 1), 1.0)
        self.fuse = init_(nn.Conv2d(2 * M, W, 1), RG)
        self.nf = norm(W)
        self.c1 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n1 = norm(W)
        self.c2 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = norm(W)
        self.raw = init_(nn.Conv2d(cin, RAW, 1), 1.0)
        hc = W + RAW
        self.proj = init_(nn.Conv2d(hc, P, 1), 1.0)
        self.fpc = nn.ModuleList([FpConv(P, HID, dy, dx, RG) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(hc, 1, 1), 0.01)
        gin = 3 * W + RAW
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        a = self.sa(obs)
        b = self.sb(obs)
        h = torch.cat([F.relu(a), a * b], dim=1)
        s = F.relu(self.nf(self.fuse(h)))
        x = F.relu(s + self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        r = self.raw(obs)
        hc = torch.cat([f, r], dim=1)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), s.amax((2, 3)), r.amax((2, 3))], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        anc = self.anchor(hc)
        p = self.proj(hc)
        res = []
        for i in range(len(self.fps)):
            z = F.relu(self.fpc[i](p))
            res.append(self.outs[i](z) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
