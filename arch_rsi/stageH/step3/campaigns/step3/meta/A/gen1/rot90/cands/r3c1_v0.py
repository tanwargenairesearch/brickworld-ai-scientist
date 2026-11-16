import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1.5e-3, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32,
          "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
WIDTH = 40
GROUPS = 8
SKIP = 16
PROJ = 32
HID = 32
GL = 96
DET = 32
DONE_SCALE = 0.1
DONE_OFFSET = -1.0


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def foot_pool(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


def add_coords(obs):
    B, _, Y, X = obs.shape
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([obs, ys, xs], dim=1)


class Block(nn.Module):
    def __init__(self, c, groups):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.norm = nn.GroupNorm(groups, c)

    def forward(self, x):
        return F.relu(x + self.norm(self.conv(x)))


class MultiScale(nn.Module):
    def __init__(self, c, groups):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = nn.GroupNorm(groups, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, c)
        self.merge = nn.Conv2d(3 * c, c, 1)
        self.n3 = nn.GroupNorm(groups, c)

    def forward(self, x):
        Y, X = x.shape[2], x.shape[3]
        h1 = (max(1, (Y + 1) // 2), max(1, (X + 1) // 2))
        h2 = (max(1, (Y + 3) // 4), max(1, (X + 3) // 4))
        a = F.adaptive_avg_pool2d(x, h1)
        a = F.relu(self.n1(self.c1(a)))
        b = F.adaptive_avg_pool2d(a, h2)
        b = F.relu(self.n2(self.c2(b)))
        au = F.interpolate(a, size=(Y, X), mode="nearest")
        bu = F.interpolate(b, size=(Y, X), mode="nearest")
        m = self.merge(torch.cat([x, au, bu], dim=1))
        return F.relu(x + self.n3(m))


class FactorHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        n = len(self.fps)
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.Conv2d(3 * proj, hid, 1)
        self.temb = nn.Parameter(torch.zeros(n, hid, 1, 1))
        self.out_shared = nn.Conv2d(hid, 1, 1)
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        self.tbias = nn.Parameter(torch.zeros(n))

    def init_out(self):
        ortho(self.out_shared, 0.01)
        ortho(self.outs, 0.01)

    def forward(self, f):
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = foot_pool(p, dy, dx)
            h = F.relu(self.mix(torch.cat([mx, mn, av], dim=1)) + self.temb[i])
            res.append(self.out_shared(h) + self.outs[i](h) + self.tbias[i])
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.Z = int(spec.Z)
        self.stem = nn.Sequential(
            nn.Conv2d(c + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.GroupNorm(GROUPS, WIDTH), nn.ReLU())
        self.trunk = nn.Sequential(Block(WIDTH, GROUPS), MultiScale(WIDTH, GROUPS), Block(WIDTH, GROUPS))
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.head = FactorHead(WIDTH + SKIP, spec.footprints, PROJ, HID)
        self.det = nn.Conv2d(2 * self.Z + 1, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * SKIP + 2 * DET, GL), nn.ReLU())
        self.done_head = nn.Linear(GL, 1)
        self.value_head = nn.Sequential(nn.Linear(GL, GL), nn.ReLU(), nn.Linear(GL, 1))
        ortho(self, math.sqrt(2.0))
        self.head.init_out()
        ortho(self.done_head, 0.01)
        ortho(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        Z = self.Z
        f = self.trunk(self.stem(add_coords(obs)))
        raw = F.relu(self.skip(obs))
        d = F.relu(self.det(torch.cat([obs[:, 2 * Z:4 * Z], obs[:, 4 * Z + 2:4 * Z + 3]], dim=1)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                                 raw.mean((2, 3)), raw.amax((2, 3)),
                                 d.mean((2, 3)), d.amax((2, 3))], dim=1))
        place = self.head(torch.cat([f, raw], dim=1)).flatten(1)
        done = DONE_SCALE * self.done_head(g) + DONE_OFFSET
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
