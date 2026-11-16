import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

COLW = 96
K = 4
LH = 8
TW = 32
PROJ = 24
HID = 24


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def fp_stats(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for fp in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for fp in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = fp_stats(p, dy, dx)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Lattice(nn.Module):
    def __init__(self, c, z, footprints):
        super().__init__()
        self.z = z
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.lat = nn.Conv2d(c, z * K, 1)
        self.mix = nn.ModuleList([nn.Conv2d(z * 2 * K, z * LH, 1, groups=z) for fp in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(z * LH, z, 1, groups=z) for fp in self.fps])

    def forward(self, h):
        b = h.shape[0]
        yy = h.shape[2]
        xx = h.shape[3]
        z = self.z
        a = self.lat(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = fp_stats(a, dy, dx)
            u = torch.cat([mn.reshape(b, z, K, yy, xx), mx.reshape(b, z, K, yy, xx)], dim=2).reshape(b, z * 2 * K, yy, xx)
            s = out(F.relu(mix(u)))
            res.append(torch.logsumexp(s, dim=1, keepdim=True))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.col = nn.Sequential(nn.Conv2d(c, COLW, 1), nn.ReLU(), nn.Conv2d(COLW, COLW, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COLW, TW, 3, padding=1)
        self.n1 = nn.GroupNorm(4, TW)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.n2 = nn.GroupNorm(4, TW)
        self.lattice = Lattice(COLW, spec.Z, spec.footprints)
        self.dense = PoolHead(COLW + TW, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * TW + COLW, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho(self, math.sqrt(2.0))
        for o in self.dense.outs:
            ortho(o, 0.01)
        for o in self.lattice.outs:
            ortho(o, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        h = self.col(obs)
        x = F.relu(self.n1(self.c1(h)))
        f = F.relu(x + self.n2(self.c2(x)))
        place = self.lattice(h) + self.dense(torch.cat([h, f], dim=1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
