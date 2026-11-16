import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 96
WIDTH = 48
PROJ = 32
HID = 32
NBLOCKS = 2


def small_init(m):
    nn.init.normal_(m.weight, std=0.01)
    nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.mix = nn.Conv2d(c, c, 1)

    def forward(self, x):
        h = self.conv(F.relu(self.norm(x)))
        return x + self.mix(F.relu(h))


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            small_init(o)

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
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1))
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(NBLOCKS)])
        self.norm = nn.GroupNorm(1, WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        small_init(self.done_head)

    def forward(self, obs, mask):
        x = self.blocks(self.stem(obs))
        f = F.relu(self.norm(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
