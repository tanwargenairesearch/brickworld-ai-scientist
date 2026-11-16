import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'n_minibatches': 64}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 16
HID = 16
GH = 64
SCALE = 5.0


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


def fp_pool(q, dy, dx):
    if dy * dx == 1:
        return q, q, q
    z = F.pad(q, (0, dx - 1, 0, dy - 1))
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
        G = self.nf
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1), math.sqrt(2))
        self.mix = nn.ModuleList([GLin(G, 3 * PROJ, HID, math.sqrt(2)) for _ in self.fps])
        self.out = nn.ModuleList([GLin(G, HID, 1, 0.01) for _ in self.fps])
        self.glob = Glob(c, self.T)
        self.alpha = nn.Parameter(torch.zeros(G))

    def forward(self, obs, mask):
        f = self.trunk(obs)
        p = self.proj(f)
        G = self.nf
        q = torch.cat([dihedral(p, k) for k in range(G)], 1)
        b, _, h, w = q.shape
        maps = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.out):
            mx, mn, av = fp_pool(q, dy, dx)
            s = torch.cat([mx.reshape(b, G, PROJ, h, w), mn.reshape(b, G, PROJ, h, w),
                           av.reshape(b, G, PROJ, h, w)], 2)
            o = out(F.relu(mix(s)))
            maps.append(o[:, :, 0])
        L = torch.stack(maps, 2)
        wt = torch.softmax(SCALE * self.alpha, 0)
        place = (L * wt.view(1, G, 1, 1, 1)).sum(1)
        g, done, value, tb = self.glob(f)
        place = place + tb.view(-1, self.T, 1, 1)
        logits = torch.cat([place.flatten(1), done], 1)
        return logits, value


def build_policy(spec):
    return Net(spec)
