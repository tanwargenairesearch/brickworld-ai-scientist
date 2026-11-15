import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
CRIT = 32
GROUPS = 8
R2 = math.sqrt(2.0)
DONE_PRIOR = -3.0


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = init_layer(nn.Conv2d(c, proj, 1), R2)
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(3 * proj, hid, 1), R2) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(hid, 1, 1), 0.01) for _ in self.fps])

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
        cin = spec.in_channels
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(cin, STEM, 1), R2), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, COL, 1), R2), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(GROUPS, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(GROUPS, WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, 64), R2), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        self.critic_conv = nn.Sequential(
            init_layer(nn.Conv2d(cin, CRIT, 1), R2), nn.ReLU(),
            init_layer(nn.Conv2d(CRIT, CRIT, 3, padding=1), R2), nn.ReLU())
        self.critic_mlp = nn.Sequential(
            init_layer(nn.Linear(2 * CRIT, 64), R2), nn.ReLU(),
            init_layer(nn.Linear(64, 1), 1.0))

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g) + DONE_PRIOR
        logits = torch.cat([place, done], dim=1)
        c = self.critic_conv(obs)
        v = self.critic_mlp(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1)).squeeze(1)
        return logits, v


def build_policy(spec):
    return Net(spec)
