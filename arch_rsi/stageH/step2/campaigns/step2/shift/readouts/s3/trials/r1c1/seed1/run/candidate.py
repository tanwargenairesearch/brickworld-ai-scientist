import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 40
WIDTH = 40
PROJ = 32
HID = 32
GROUPS = 4


class TokenMix(nn.Module):
    def __init__(self, c, n, groups):
        super().__init__()
        self.g = groups
        self.w = nn.Parameter(torch.empty(groups, n, n))
        self.b = nn.Parameter(torch.zeros(groups, 1, n))
        bound = 1.0 / math.sqrt(n)
        nn.init.uniform_(self.w, -bound, bound)

    def forward(self, f):
        B, C, Y, X = f.shape
        t = f.reshape(B, self.g, C // self.g, Y * X)
        out = torch.einsum('bgcn,gmn->bgcm', t, self.w) + self.b.unsqueeze(0)
        return out.reshape(B, C, Y, X)


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
        n = spec.Y * spec.X
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.mix1 = TokenMix(COL, n, GROUPS)
        self.fuse1 = nn.Conv2d(2 * COL, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.mix2 = TokenMix(WIDTH, n, GROUPS)
        self.fuse2 = nn.Conv2d(2 * WIDTH, WIDTH, 1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        s = F.relu(s + self.fuse1(torch.cat([s, self.mix1(s)], dim=1)))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        f = F.relu(f + self.fuse2(torch.cat([f, self.mix2(f)], dim=1)))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
