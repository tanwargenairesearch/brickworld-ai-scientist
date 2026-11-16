import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

W = 40
RAW = 16
P = 16
Q = 8
HID = 32
CNT = 16
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
        self.shapes = sorted(set(self.fps))
        self.stem1 = init_(nn.Conv2d(cin, 48, 1), RG)
        self.ns = norm(48)
        self.stem2 = init_(nn.Conv2d(48, W, 1), RG)
        self.c1 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n1 = norm(W)
        self.c2 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = norm(W)
        self.raw = init_(nn.Conv2d(cin, RAW, 1), 1.0)
        self.cnt = init_(nn.Conv2d(cin, CNT, 1), RG)
        hc = W + RAW
        self.proj = init_(nn.Conv2d(hc, P, 1), 1.0)
        self.qproj = init_(nn.Conv2d(hc, Q, 1), 1.0)
        self.fpc = nn.ModuleList([FpConv(P, HID, dy, dx, RG) for dy, dx in self.fps])
        self.pmix = nn.ModuleList([init_(nn.Conv2d(3 * Q * len(self.shapes), HID, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(hc, 1, 1), 0.01)
        gin = 2 * W + RAW + CNT
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.dense = init_(nn.Linear(G, int(spec.n_place)), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.ns(self.stem1(obs)))
        s = F.relu(self.stem2(s))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        r = self.raw(obs)
        hc = torch.cat([f, r], dim=1)
        cnt = torch.log1p(F.relu(self.cnt(obs)).sum((2, 3)))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), r.amax((2, 3)), cnt], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        anc = self.anchor(hc)
        p = self.proj(hc)
        q = self.qproj(hc)
        parts = []
        for a, b in self.shapes:
            parts = parts + fp_pool(q, a, b)
        pooled = torch.cat(parts, dim=1)
        res = []
        for i in range(len(self.fps)):
            z = F.relu(self.fpc[i](p) + self.pmix[i](pooled))
            res.append(self.outs[i](z) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
