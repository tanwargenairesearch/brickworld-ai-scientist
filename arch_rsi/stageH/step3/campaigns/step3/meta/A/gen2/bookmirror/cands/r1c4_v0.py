import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 16
HID = 16
W3 = 24
GH = 64
SCALE = 5.0
CORNERS = ((True, True), (True, False), (False, True), (False, False))


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def wpool(q, dy, dx, top=True, left=True):
    if dy * dx == 1:
        return q, q, q
    pad = ((0, dx - 1) if left else (dx - 1, 0)) + ((0, dy - 1) if top else (dy - 1, 0))
    q = F.pad(q, pad)
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Base(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.s1 = ortho(nn.Conv2d(c, STEM, 1), math.sqrt(2))
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), math.sqrt(2))
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), math.sqrt(2))
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), math.sqrt(2))
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), 1.0)
        self.ch = WIDTH + SKIP
        self.glob = nn.Sequential(ortho(nn.Linear(2 * self.ch, GH), math.sqrt(2)), nn.ReLU(),
                                  ortho(nn.Linear(GH, GH), math.sqrt(2)), nn.ReLU())
        self.done = ortho(nn.Linear(GH, 1), 0.01)
        self.value = ortho(nn.Linear(GH, 1), 1.0)
        self.tbias = ortho(nn.Linear(GH, spec.n_types), 0.01)

    def features(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, self.skip(obs)], 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        return f, g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.base = Base(spec)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        nc = len(CORNERS)
        self.nc = nc
        self.proj = ortho(nn.Conv2d(self.base.ch, PROJ, 1), math.sqrt(2))
        self.mix = nn.ModuleList([ortho(nn.Conv2d(nc * 3 * PROJ, nc * HID, 1, groups=nc), math.sqrt(2)) for _ in self.fps])
        self.out = nn.ModuleList([ortho(nn.Conv2d(nc * HID, nc, 1, groups=nc), 0.01) for _ in self.fps])
        self.c3 = ortho(nn.Conv2d(self.base.ch, W3, 3, padding=1), math.sqrt(2))
        self.o3 = ortho(nn.Conv2d(W3, self.T, 1), 0.01)
        self.alpha = nn.Parameter(torch.zeros(nc + 1))

    def forward(self, obs, mask):
        f, g = self.base.features(obs)
        p = self.proj(f)
        B = p.shape[0]
        per_t = []
        for t, (dy, dx) in enumerate(self.fps):
            zs = []
            for top, left in CORNERS:
                mx, mn, av = wpool(p, dy, dx, top, left)
                zs.append(torch.cat([mx, mn, av], 1))
            h = F.relu(self.mix[t](torch.cat(zs, 1)))
            per_t.append(self.out[t](h))
        Lc = torch.stack(per_t, 2)
        L3 = self.o3(F.relu(self.c3(f))).unsqueeze(1)
        L = torch.cat([Lc, L3], 1).reshape(B, self.nc + 1, -1)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = (L * w.view(1, self.nc + 1, 1)).sum(1)
        n = place.shape[1] // self.T
        place = place + self.base.tbias(g).unsqueeze(2).expand(B, self.T, n).reshape(B, -1)
        logits = torch.cat([place, self.base.done(g)], 1)
        return logits, self.base.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
