import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 1e-3, 'n_envs': 256, 'rollout_len': 16, 'n_minibatches': 16, 'ppo_epochs': 3, 'ent_coef': 0.005, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.3}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
PROJ = 48
HID = 32
GROUPS = 4


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(GROUPS, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(GROUPS, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        n = len(self.fps)
        self.proj = nn.Conv2d(c, proj, 1)
        self.foot = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1, bias=False) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        self.inter = nn.Conv2d(c, n, 1)
        self.anchor = nn.Conv2d(c, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(n))

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), foot, mix, out in zip(self.fps, self.foot, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(foot(q) + mix(torch.cat([mx, mn], dim=1)))
            res.append(out(h))
        s = torch.cat(res, dim=1) + self.inter(f) + self.anchor(f)
        s = s + self.tbias.view(1, -1, 1, 1)
        return s.flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.GroupNorm(GROUPS, STEM), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.inp = nn.Conv2d(COL, WIDTH, 1)
        self.res = Res(WIDTH)
        self.norm = nn.GroupNorm(GROUPS, WIDTH)
        c = WIDTH + SKIP
        self.head = FootHead(c, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        for m in self.head.outs:
            init_layer(m, 0.01)
        init_layer(self.head.inter, 0.01)
        init_layer(self.head.anchor, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        s = self.stem(obs)
        f = F.relu(self.norm(self.res(self.inp(s))))
        k = F.relu(self.skip(obs))
        h = torch.cat([f, k], dim=1)
        place = self.head(h)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
