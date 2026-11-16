import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 8, "n_minibatches": 8, "ppo_epochs": 4}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
MIX = 24


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
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        C = spec.in_channels
        self.rowmix = nn.Conv2d(C, MIX, (1, self.X))
        self.colmix = nn.Conv2d(C, MIX, (self.Y, 1))
        self.stem = nn.Sequential(
            nn.Conv2d(C + 2 * MIX, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        X = self.X
        Y = self.Y
        pr = F.pad(obs, (X // 2, X - 1 - X // 2, 0, 0), mode='circular')
        r = self.rowmix(pr)
        pc = F.pad(obs, (0, 0, Y // 2, Y - 1 - Y // 2), mode='circular')
        c = self.colmix(pc)
        h = self.stem(torch.cat([obs, r, c], dim=1))
        x = F.relu(self.c1(h))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
