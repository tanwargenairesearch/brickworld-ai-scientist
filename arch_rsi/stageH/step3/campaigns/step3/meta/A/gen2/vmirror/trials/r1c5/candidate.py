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


def stack3(mx, mn, av, g):
    b, c, h, w = mx.shape
    s = torch.stack([mx.reshape(b, g, c // g, h, w), mn.reshape(b, g, c // g, h, w),
                     av.reshape(b, g, c // g, h, w)], 2)
    return s.reshape(b, 3 * c, h, w)


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
        T = self.T
        G = 2
        self.G = G
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1), math.sqrt(2))
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(G * 3 * PROJ, G * T * HID, 1, groups=G), math.sqrt(2)) for _ in self.fps])
        self.out = nn.ModuleList([init_layer(nn.Conv2d(G * T * HID, G * T, 1, groups=G * T), 0.01) for _ in self.fps])
        self.glob = Glob(c, T)
        self.alpha = nn.Parameter(torch.zeros(G))
        self.assign = nn.Parameter(torch.zeros(T, T))

    def forward(self, obs, mask):
        f = self.trunk(obs)
        p = self.proj(f)
        if self.Y == self.X:
            p2 = p.transpose(2, 3)
        else:
            p2 = p.flip(2).flip(3)
        q = torch.cat([p, p2], 1)
        G, T = self.G, self.T
        maps = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.out):
            mx, mn, av = fp_pool(q, dy, dx)
            h = F.relu(mix(stack3(mx, mn, av, G)))
            o = out(h)
            maps.append(o.reshape(o.shape[0], G, T, o.shape[2], o.shape[3]))
        L = torch.stack(maps, 3)
        wf = torch.softmax(SCALE * self.alpha, 0)
        A = torch.softmax(SCALE * self.assign, 1)
        place = torch.einsum('g,tf,bgtfyx->btyx', wf, A, L)
        g, done, value, tb = self.glob(f)
        place = place + tb.view(-1, T, 1, 1)
        logits = torch.cat([place.flatten(1), done], 1)
        return logits, value


def build_policy(spec):
    return Net(spec)
