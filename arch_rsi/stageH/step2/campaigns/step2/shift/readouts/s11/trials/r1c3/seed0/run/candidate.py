import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.02}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


def axis_param(c, n):
    w = nn.Parameter(torch.empty(c, n, n))
    b = 1.0 / math.sqrt(n)
    nn.init.uniform_(w, -b, b)
    return w


class AxisMix(nn.Module):
    def __init__(self, c, ny, nx):
        super().__init__()
        self.wx = axis_param(c, nx)
        self.wy = axis_param(c, ny)
        self.wx2 = axis_param(c, nx)
        self.wy2 = axis_param(c, ny)

    def forward(self, o):
        mx = torch.einsum('bcyx,cvx->bcyv', o, self.wx)
        my = torch.einsum('bcyx,cuy->bcux', o, self.wy)
        mxy = torch.einsum('bcyx,cvx->bcyv', o, self.wx2)
        mxy = torch.einsum('bcyx,cuy->bcux', mxy, self.wy2)
        return torch.cat([o, mx, my, mxy], dim=1)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.amix = AxisMix(c, spec.Y, spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(4 * c, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(self.amix(obs))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
