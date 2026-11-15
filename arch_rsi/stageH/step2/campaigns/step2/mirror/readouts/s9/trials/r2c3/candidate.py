import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 32

ORIENTS = [(False, False), (False, True), (True, False), (True, True)]


class OrientPoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        n_in = 3 * proj * len(ORIENTS)
        self.mix = nn.ModuleList([nn.Conv2d(n_in, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def pooled(self, p, dy, dx, fy, fx):
        if dy * dx == 1:
            t = torch.cat([p, p, p], dim=1)
        else:
            left = dx - 1 if fx else 0
            right = 0 if fx else dx - 1
            top = dy - 1 if fy else 0
            bottom = 0 if fy else dy - 1
            q = F.pad(p, (left, right, top, bottom))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
            t = torch.cat([mx, mn, av], dim=1)
        if fy and fx:
            t = torch.flip(t, dims=[2, 3])
        elif fy:
            t = torch.flip(t, dims=[2])
        elif fx:
            t = torch.flip(t, dims=[3])
        return t

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            feats = [self.pooled(p, dy, dx, fy, fx) for fy, fx in ORIENTS]
            h = F.relu(mix(torch.cat(feats, dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = OrientPoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
