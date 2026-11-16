import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 8
HID = 8
GH = 64
SCALE = 5.0
CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class GLin(nn.Module):
    def __init__(self, G, cin, cout, gain):
        super().__init__()
        w = torch.empty(G, cout, cin)
        for g in range(G):
            nn.init.orthogonal_(w[g], gain)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(G, cout))

    def forward(self, x):
        G = self.w.shape[0]
        return torch.einsum('bgchw,gdc->bgdhw', x, self.w) + self.b.view(1, G, -1, 1, 1)


def dihedral(t, k):
    if k & 4:
        t = t.transpose(2, 3)
    if k & 1:
        t = t.flip(2)
    if k & 2:
        t = t.flip(3)
    return t


def fp_pool(q, dy, dx, cy=0, cx=0):
    if dy * dx == 1:
        return q, q, q
    pad = (cx * (dx - 1), (1 - cx) * (dx - 1), cy * (dy - 1), (1 - cy) * (dy - 1))
    z = F.pad(q, pad)
    mx = F.max_pool2d(z, (dy, dx), stride=1)
    mn = -F.max_pool2d(-z, (dy, dx), stride=1)
    av = F.avg_pool2d(z, (dy, dx), stride=1)
    return mx, mn, av


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        g = math.sqrt(2)
        self.s1 = init_layer(nn.Conv2d(cin, STEM, 1), g)
        self.s2 = init_layer(nn.Conv2d(STEM, COL, 1), g)
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1), g)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = init_layer(nn.Conv2d(cin, SKIP, 1), g)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1)


class Glob(nn.Module):
    def __init__(self, c, T):
        super().__init__()
        g = math.sqrt(2)
        self.l1 = init_layer(nn.Linear(2 * c, GH), g)
        self.l2 = init_layer(nn.Linear(GH, GH), g)
        self.done = init_layer(nn.Linear(GH, 1), 0.01)
        self.value = init_layer(nn.Linear(GH, 1), 1.0)
        self.tb = init_layer(nn.Linear(GH, T), 0.01)

    def forward(self, f):
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        g = F.relu(self.l2(F.relu(self.l1(g))))
        return g, self.done(g), self.value(g).squeeze(1), self.tb(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.nf = 8 if self.Y == self.X else 4
        self.nc = len(CORNERS)
        G = self.nf * self.nc
        self.G = G
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1), math.sqrt(2))
        self.mix = nn.ModuleList([GLin(G, 3 * PROJ, HID, math.sqrt(2)) for _ in self.fps])
        self.out = nn.ModuleList([GLin(G, HID, 2, 0.01) for _ in self.fps])
        self.glob = Glob(c, self.T)
        self.alpha = nn.Parameter(torch.zeros(G))
        self.beta = nn.Parameter(torch.zeros(2))

    def forward(self, obs, mask):
        f = self.trunk(obs)
        p = self.proj(f)
        nf, nc, G, T = self.nf, self.nc, self.G, self.T
        q = torch.cat([dihedral(p, k) for k in range(nf)], 1)
        b, _, h, w = q.shape
        maps = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.out):
            per_corner = []
            for cy, cx in CORNERS:
                mx, mn, av = fp_pool(q, dy, dx, cy, cx)
                per_corner.append(torch.cat([mx.reshape(b, nf, PROJ, h, w), mn.reshape(b, nf, PROJ, h, w),
                                             av.reshape(b, nf, PROJ, h, w)], 2))
            s = torch.stack(per_corner, 2).reshape(b, G, 3 * PROJ, h, w)
            o = out(F.relu(mix(s)))
            maps.append(o)
        L = torch.stack(maps, 3)
        wc = torch.softmax(SCALE * self.alpha, 0)
        P2 = (L * wc.view(1, G, 1, 1, 1, 1)).sum(1)
        g, done, value, tb = self.glob(f)
        P2 = P2 + tb.view(-1, 1, T, 1, 1)
        o0 = P2[:, 0].flatten(1)
        o1 = P2[:, 1].permute(0, 2, 3, 1).flatten(1)
        wo = torch.softmax(SCALE * self.beta, 0)
        place = wo[0] * o0 + wo[1] * o1
        logits = torch.cat([place, done], 1)
        return logits, value


def build_policy(spec):
    return Net(spec)
