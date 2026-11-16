import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "n_minibatches": 16}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
AX = 40
AX2 = 28


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
        self.X = spec.X
        self.Y = spec.Y
        c = spec.in_channels
        self.s1 = nn.Conv2d(c, STEM, 1)
        self.xconv = nn.Conv2d(c, AX, (1, spec.X))
        self.yconv = nn.Conv2d(c, AX, (spec.Y, 1))
        self.s2 = nn.Conv2d(STEM + 2 * AX, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.xconv2 = nn.Conv2d(WIDTH, AX2, (1, spec.X))
        self.yconv2 = nn.Conv2d(WIDTH, AX2, (spec.Y, 1))
        self.rmix = nn.Conv2d(2 * AX2, WIDTH, 1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        a = F.relu(self.s1(obs))
        px = F.pad(obs, (0, self.X - 1, 0, 0), mode='circular')
        rx = F.relu(self.xconv(px))
        py = F.pad(obs, (0, 0, 0, self.Y - 1), mode='circular')
        ry = F.relu(self.yconv(py))
        s = F.relu(self.s2(torch.cat([a, rx, ry], dim=1)))
        x = F.relu(self.c1(s))
        qx = F.pad(x, (0, self.X - 1, 0, 0), mode='circular')
        qy = F.pad(x, (0, 0, 0, self.Y - 1), mode='circular')
        r2 = torch.cat([F.relu(self.xconv2(qx)), F.relu(self.yconv2(qy))], dim=1)
        x = F.relu(x + self.rmix(r2))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
