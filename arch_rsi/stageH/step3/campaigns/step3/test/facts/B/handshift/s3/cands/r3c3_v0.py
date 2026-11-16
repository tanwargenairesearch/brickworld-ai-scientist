import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 48
GLOB = 64
ORDERS = ['tyx', 'txy', 'ytx', 'yxt', 'xty', 'xyt']
FLIPS = [(False, False), (True, False), (False, True), (True, True)]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.sizes = {'t': self.T, 'y': self.Y, 'x': self.X}
        self.lays = [(o, fy, fx) for o in ORDERS for (fy, fx) in FLIPS]
        L = len(self.lays)
        cin = int(spec.in_channels) + 2 * L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.outs_a = nn.ModuleList([nn.Conv2d(WIDTH, self.T, 1) for _ in self.lays])
        self.outs_b = nn.ModuleList([nn.Conv2d(WIDTH, self.T, 1) for _ in self.lays])
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer('coords', torch.cat([ys, xs], 1).contiguous())
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_a = nn.Linear(GLOB, 1)
        self.done_b = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def to_grid(self, mf, order, fy, fx):
        B = mf.shape[0]
        g = mf.reshape([B] + [self.sizes[c] for c in order])
        g = g.permute([0] + [1 + order.index(c) for c in 'tyx'])
        if fy:
            g = g.flip(-2)
        if fx:
            g = g.flip(-1)
        return g

    def to_flat(self, g, order, fy, fx):
        B = g.shape[0]
        if fy:
            g = g.flip(-2)
        if fx:
            g = g.flip(-1)
        return g.permute([0] + [1 + 'tyx'.index(c) for c in order]).reshape(B, -1)

    def place(self, f, outs):
        p = None
        for (o, fy, fx), out in zip(self.lays, outs):
            q = self.to_flat(out(f), o, fy, fx)
            p = q if p is None else p + q
        return p

    def forward(self, obs, mask):
        B = obs.shape[0]
        ma = mask[:, :self.n_place].float()
        mb = mask[:, 1:self.n_place + 1].float()
        ms = [self.to_grid(ma, o, fy, fx) for (o, fy, fx) in self.lays]
        ms = ms + [self.to_grid(mb, o, fy, fx) for (o, fy, fx) in self.lays]
        c = self.coords.expand(B, -1, -1, -1)
        h = torch.cat([obs] + ms + [c], 1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        pa = self.place(f, self.outs_a)
        pb = self.place(f, self.outs_b)
        g2 = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([pa, self.done_a(g2)], dim=1) + torch.cat([self.done_b(g2), pb], dim=1)
        return logits, self.value_head(g2).squeeze(1)


def build_policy(spec):
    return Net(spec)
