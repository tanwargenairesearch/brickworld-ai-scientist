import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.01}

STEM = 64
WIDTH = 32
PROJ = 32
HID = 32
GLOB = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.g = nn.Linear(2 * c, c)

    def forward(self, x):
        g = self.g(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        h = F.relu(self.c1(x) + g[:, :, None, None])
        return F.relu(x + self.c2(h))


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid, gdim):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.gmix = nn.ModuleList([nn.Linear(gdim, hid) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f, g):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, gm, out in zip(self.fps, self.mix, self.gmix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)) + gm(g)[:, :, None, None])
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.b1 = Block(WIDTH)
        self.b2 = Block(WIDTH)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID, GLOB)
        self.done_head = nn.Linear(GLOB, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        f = self.b2(self.b1(s))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(f, g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
