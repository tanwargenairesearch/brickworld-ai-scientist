import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 8, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95, "clip_eps": 0.2}

W = 48
NB = 3
RAW = 16
P = 16
HID = 24
G = 96
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
        self.stem1 = init_(nn.Conv2d(cin, 64, 1), RG)
        self.ns = norm(64)
        self.stem2 = init_(nn.Conv2d(64, W, 1), RG)
        self.convs = nn.ModuleList([init_(nn.Conv2d(W, W, 3, padding=1), RG) for _ in range(NB)])
        self.norms = nn.ModuleList([norm(W) for _ in range(NB)])
        self.raw = init_(nn.Conv2d(cin, RAW, 1), 1.0)
        hc = W + RAW
        self.proj = init_(nn.Conv2d(hc, P, 1), 1.0)
        self.fpc = nn.ModuleList([FpConv(P, HID, dy, dx, RG) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(hc, 1, 1), 0.01)
        gin = 2 * W + 2 * RAW
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.dense = init_(nn.Linear(G, int(spec.n_place)), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.ns(self.stem1(obs)))
        x = F.relu(self.stem2(s))
        for conv, nrm in zip(self.convs, self.norms):
            x = F.relu(x + nrm(conv(x)))
        r = self.raw(obs)
        hc = torch.cat([x, r], dim=1)
        g = torch.cat([x.mean((2, 3)), x.amax((2, 3)), r.amax((2, 3)), F.relu(r).mean((2, 3))], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        anc = self.anchor(hc)
        p = self.proj(hc)
        res = []
        for i in range(len(self.fps)):
            z = F.relu(self.fpc[i](p))
            res.append(self.outs[i](z) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
