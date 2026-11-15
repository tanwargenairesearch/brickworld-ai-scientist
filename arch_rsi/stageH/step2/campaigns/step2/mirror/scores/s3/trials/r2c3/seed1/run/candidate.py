import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ppo_epochs": 2, "lr": 0.002, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 32
G = 64


def shift(u, oy, ox):
    if oy == 0 and ox == 0:
        return u
    Y = u.shape[2]
    X = u.shape[3]
    q = F.pad(u, (0, ox, 0, oy))
    return q[:, :, oy:oy + Y, ox:ox + X]


class RelHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        n = len(self.fps)
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.cell = nn.ModuleList([nn.Conv2d(c, HID, 1) for _ in range(n)])
        self.ctx = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in range(n)])
        self.mid = nn.ModuleList([nn.Conv2d(2 * HID, HID, 1) for _ in range(n)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(n)])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            c = self.ctx[i](torch.cat([mx, mn, av], dim=1))
            u = self.cell[i](f)
            hs = []
            for oy in range(dy):
                for ox in range(dx):
                    hs.append(F.relu(shift(u, oy, ox) + c))
            hst = torch.stack(hs, 0)
            agg = torch.cat([hst.amin(0), hst.mean(0)], dim=1)
            h = F.relu(self.mid[i](agg))
            res.append(self.outs[i](h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        fc = COL + WIDTH
        self.head = RelHead(fc, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * fc, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        fc = torch.cat([s, f], dim=1)
        place = self.head(fc)
        g = self.glob(torch.cat([fc.mean((2, 3)), fc.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
