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
WIN = 16
GH = 64
SCALE = 5.0
CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


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
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        G = len(CORNERS)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1), math.sqrt(2))
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(G * 3 * PROJ, G * HID, 1, groups=G), math.sqrt(2)) for _ in self.fps])
        self.out = nn.ModuleList([init_layer(nn.Conv2d(G * HID, G, 1, groups=G), 0.01) for _ in self.fps])
        self.win = init_layer(nn.Conv2d(c, WIN, 3, padding=1), math.sqrt(2))
        self.wout = init_layer(nn.Conv2d(WIN, self.T, 1), 0.01)
        self.glob = Glob(c, self.T)
        self.alpha = nn.Parameter(torch.zeros(G + 1))

    def forward(self, obs, mask):
        f = self.trunk(obs)
        p = self.proj(f)
        maps = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.out):
            parts = []
            for cy, cx in CORNERS:
                mx, mn, av = fp_pool(p, dy, dx, cy, cx)
                parts += [mx, mn, av]
            h = F.relu(mix(torch.cat(parts, 1)))
            maps.append(out(h))
        Lc = torch.stack(maps, 2)
        wo = self.wout(F.relu(self.win(f))).unsqueeze(1)
        L = torch.cat([Lc, wo], 1)
        n = L.shape[1]
        w = torch.softmax(SCALE * self.alpha, 0)
        place = (L * w.view(1, n, 1, 1, 1)).sum(1)
        g, done, value, tb = self.glob(f)
        place = place + tb.view(-1, self.T, 1, 1)
        logits = torch.cat([place.flatten(1), done], 1)
        return logits, value


def build_policy(spec):
    return Net(spec)
