import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1e-3, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32,
          "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
WIDTH = 32
GROUPS = 8
SKIP = 16
PROJ = 24
HID = 32
GL = 96
DET = 32
MAX_ITERS = 6


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


class IterTrunk(nn.Module):
    def __init__(self, c, groups, iters):
        super().__init__()
        self.iters = iters
        self.inj = nn.Conv2d(c, c, 1)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.norm = nn.GroupNorm(groups, c)

    def forward(self, e):
        a = self.inj(e)
        x = e
        for k in range(self.iters):
            x = F.relu(x + self.norm(self.conv(x) + a))
        return x


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
        iters = min(MAX_ITERS, max(3, (max(int(spec.Y), int(spec.X)) + 1) // 2 + 1))
        self.stem = nn.Sequential(
            nn.Conv2d(c + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.GroupNorm(GROUPS, WIDTH), nn.ReLU())
        self.trunk = IterTrunk(WIDTH, GROUPS, iters)
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
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
