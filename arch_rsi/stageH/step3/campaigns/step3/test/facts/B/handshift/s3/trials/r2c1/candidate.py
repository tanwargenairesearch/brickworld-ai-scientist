import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 48
GLOB = 64
ORDERS = ['tyx', 'txy', 'yxt', 'xyt']
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
        cin = int(spec.in_channels) + L * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.outs = nn.ModuleList([nn.Conv2d(WIDTH, self.T, 1) for _ in self.lays])
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer('coords', torch.cat([ys, xs], 1).contiguous())
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def to_grid(self, mf, order):
        B = mf.shape[0]
        g = mf.reshape([B] + [self.sizes[c] for c in order])
        return g.permute([0] + [1 + order.index(c) for c in 'tyx'])

    def to_flat(self, g, order):
        B = g.shape[0]
        return g.permute([0] + [1 + 'tyx'.index(c) for c in order]).reshape(B, -1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        mf = mask[:, :self.n_place].float()
        ms = []
        for (o, fy, fx) in self.lays:
            g = self.to_grid(mf, o)
            if fy:
                g = g.flip(-2)
            if fx:
                g = g.flip(-1)
            ms.append(g)
        c = self.coords.expand(B, -1, -1, -1)
        h = torch.cat([obs] + ms + [c], 1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = None
        for (o, fy, fx), out in zip(self.lays, self.outs):
            g = out(f)
            if fy:
                g = g.flip(-2)
            if fx:
                g = g.flip(-1)
            p = self.to_flat(g, o)
            place = p if place is None else place + p
        g2 = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g2)], dim=1)
        return logits, self.value_head(g2).squeeze(1)


def build_policy(spec):
    return Net(spec)
