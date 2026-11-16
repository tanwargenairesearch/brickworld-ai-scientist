import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

STEM = 64
W = 48
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
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem1 = init_(nn.Conv2d(cin, STEM, 1), RG)
        self.stem2 = init_(nn.Conv2d(STEM, W, 1), RG)
        self.c1 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n1 = norm(W)
        self.c2 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = norm(W)
        self.proj = init_(nn.Conv2d(W, P, 1), 1.0)
        self.loc = nn.ModuleList([init_(nn.Conv2d(P, P, 3, padding=1), 1.0) for _ in self.fps])
        self.mix = nn.ModuleList([init_(nn.Conv2d(4 * P, HID, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.gnorm = nn.LayerNorm(2 * W)
        self.glob = init_(nn.Linear(2 * W, G), RG)
        self.dense = init_(nn.Linear(G, spec.n_place), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat(fp_pool(p, dy, dx) + [self.loc[i](p)], dim=1)
            z = F.relu(self.mix[i](z))
            res.append(self.outs[i](z))
        place = torch.cat(res, dim=1).flatten(1) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
