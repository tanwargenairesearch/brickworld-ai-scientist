import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

S = 24
W = 40
RAW = 16
Q0 = 16
P = 24
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


def dihedral(t, square):
    outs = [t, t.flip(-1), t.flip(-2), t.flip(-1).flip(-2)]
    if square:
        u = t.transpose(-1, -2)
        outs = outs + [u, u.flip(-1), u.flip(-2), u.flip(-1).flip(-2)]
    return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.sq = int(spec.Y) == int(spec.X)
        nd = 8 if self.sq else 4
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.stem1 = init_(nn.Conv2d(cin, 48, 1), RG)
        self.stem2 = init_(nn.Conv2d(48, S, 1), RG)
        self.dmix = init_(nn.Conv2d(S * nd, W, 1), RG)
        self.n0 = norm(W)
        self.c1 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n1 = norm(W)
        self.c2 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = norm(W)
        self.raw = init_(nn.Conv2d(cin, RAW, 1), 1.0)
        self.q0 = init_(nn.Conv2d(W + RAW, Q0, 1), 1.0)
        self.qmix = init_(nn.Conv2d(Q0 * nd, P, 1), 1.0)
        self.shapes = []
        mixes = []
        for dy, dx in self.fps:
            sh = [(dy, dx)] if dy == dx else [(dy, dx), (dx, dy)]
            self.shapes.append(sh)
            mixes.append(init_(nn.Conv2d(3 * P * len(sh), HID, 1), RG))
        self.mix = nn.ModuleList(mixes)
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(P, 1, 1), 0.01)
        gin = 2 * W + RAW
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n0(self.dmix(dihedral(s, self.sq))))
        h = F.relu(self.n1(self.c1(x)))
        f = F.relu(x + self.n2(self.c2(h)))
        r = self.raw(obs)
        q = self.q0(torch.cat([f, r], dim=1))
        p = self.qmix(dihedral(q, self.sq))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), r.amax((2, 3))], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        anc = self.anchor(p)
        res = []
        for i in range(len(self.fps)):
            parts = []
            for a, b in self.shapes[i]:
                parts = parts + fp_pool(p, a, b)
            z = F.relu(self.mix[i](torch.cat(parts, dim=1)))
            res.append(self.outs[i](z) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
