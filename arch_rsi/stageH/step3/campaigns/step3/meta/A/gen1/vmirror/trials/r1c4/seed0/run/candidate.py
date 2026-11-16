import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

COLW = 96
K = 4
LH = 8
TW = 32
SKIP = 16
PROJ = 24
HID = 24


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def add_coords(obs):
    b = obs.shape[0]
    yy = obs.shape[2]
    xx = obs.shape[3]
    cy = torch.linspace(-1.0, 1.0, yy, device=obs.device, dtype=obs.dtype).view(1, 1, yy, 1).expand(b, 1, yy, xx)
    cx = torch.linspace(-1.0, 1.0, xx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, xx).expand(b, 1, yy, xx)
    return torch.cat([obs, cy, cx], dim=1)


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
        return torch.cat(res, dim=1)


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
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels + 2
        self.col = nn.Sequential(nn.Conv2d(c, COLW, 1), nn.ReLU(), nn.Conv2d(COLW, COLW, 1), nn.ReLU())
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.c1 = nn.Conv2d(COLW, TW, 3, padding=1)
        self.n1 = nn.GroupNorm(4, TW)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.n2 = nn.GroupNorm(4, TW)
        self.lattice = Lattice(COLW, spec.Z, spec.footprints)
        self.dense = PoolHead(COLW + TW + SKIP, spec.footprints, PROJ, HID)
        self.anchor = nn.Conv2d(COLW + TW + SKIP, 1, 1)
        self.glob = nn.Sequential(nn.Linear(2 * TW + COLW + SKIP, 64), nn.ReLU())
        self.type_head = nn.Linear(64, spec.n_types)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho(self, math.sqrt(2.0))
        for o in self.dense.outs:
            ortho(o, 0.01)
        for o in self.lattice.outs:
            ortho(o, 0.01)
        ortho(self.anchor, 0.01)
        ortho(self.type_head, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        o = add_coords(obs)
        h = self.col(o)
        k = F.relu(self.skip(o))
        x = F.relu(self.n1(self.c1(h)))
        f = F.relu(x + self.n2(self.c2(x)))
        hf = torch.cat([h, f, k], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h.amax((2, 3)), k.amax((2, 3))], dim=1))
        typ = self.type_head(g)
        place = self.lattice(h) + self.dense(hf) + self.anchor(hf) + typ[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
