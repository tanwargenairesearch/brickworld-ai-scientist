import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
SKIP = 16
GC = 32
DONE_BIAS = -2.0


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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.head = PoolHead(WIDTH + SKIP, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * (WIDTH + SKIP), 64), nn.ReLU())
        self.gcell = nn.Conv2d(c, GC, 1)
        self.gate = nn.Sequential(nn.Linear(2 * GC + 64, 64), nn.ReLU())
        self.gate_out = nn.Linear(64, 1)
        self.value_head = nn.Linear(128, 1)
        ortho(self, math.sqrt(2.0))
        for o in self.head.outs:
            ortho(o, 0.01)
        ortho(self.gate_out, 0.01)
        nn.init.constant_(self.gate_out.bias, DONE_BIAS)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = self.stem(obs)
        k = F.relu(self.skip(obs))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        h = torch.cat([f, k], dim=1)
        place = self.head(h)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        m = F.relu(self.gcell(obs))
        q = self.gate(torch.cat([m.amax((2, 3)), m.mean((2, 3)), g], dim=1))
        done = self.gate_out(q) + torch.logsumexp(place, dim=1, keepdim=True).detach()
        logits = torch.cat([place, done], dim=1)
        value = self.value_head(torch.cat([g, q], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
