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
GH = 64
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def dihedral(x, k):
    if k >= 4:
        x = x.transpose(-2, -1)
    m = k % 4
    if m == 1:
        x = x.flip(-1)
    elif m == 2:
        x = x.flip(-2)
    elif m == 3:
        x = x.flip(-2).flip(-1)
    return x


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
        self.frames = list(range(8)) if spec.X == spec.Y else [0, 1, 2, 3]
        nf = len(self.frames)
        self.nf = nf
        self.proj = ortho(nn.Conv2d(self.base.ch, PROJ, 1), math.sqrt(2))
        self.mix = nn.ModuleList([ortho(nn.Conv2d(nf * 3 * PROJ, nf * HID, 1, groups=nf), math.sqrt(2)) for _ in self.fps])
        self.out = nn.ModuleList([ortho(nn.Conv2d(nf * HID, nf, 1, groups=nf), 0.01) for _ in self.fps])
        self.alpha = nn.Parameter(torch.zeros(nf))

    def forward(self, obs, mask):
        f, g = self.base.features(obs)
        p = self.proj(f)
        B = p.shape[0]
        nf = self.nf
        u = torch.stack([dihedral(p, k) for k in self.frames], 1)
        Yf, Xf = u.shape[-2], u.shape[-1]
        u = u.reshape(B, nf * PROJ, Yf, Xf)
        maps = []
        for t, (dy, dx) in enumerate(self.fps):
            mx, mn, av = wpool(u, dy, dx)
            z = torch.stack([mx.reshape(B, nf, PROJ, Yf, Xf), mn.reshape(B, nf, PROJ, Yf, Xf),
                             av.reshape(B, nf, PROJ, Yf, Xf)], 2).reshape(B, nf * 3 * PROJ, Yf, Xf)
            h = F.relu(self.mix[t](z))
            maps.append(self.out[t](h))
        L = torch.stack(maps, 2).reshape(B, nf, -1)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = (L * w.view(1, nf, 1)).sum(1)
        tb = self.base.tbias(g).unsqueeze(2).expand(B, self.T, Yf * Xf).reshape(B, -1)
        place = place + tb
        logits = torch.cat([place, self.base.done(g)], 1)
        return logits, self.base.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
