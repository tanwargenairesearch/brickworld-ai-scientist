import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 48
GLOB = 64


def make_layouts(square):
    lay = [(False, False, False), (False, True, False), (False, False, True)]
    if square:
        lay.insert(1, (True, False, False))
    return lay


def fwd_t(x, t, fy, fx):
    if t:
        x = x.transpose(-1, -2)
    if fy:
        x = x.flip(-2)
    if fx:
        x = x.flip(-1)
    return x


def inv_t(x, t, fy, fx):
    if fx:
        x = x.flip(-1)
    if fy:
        x = x.flip(-2)
    if t:
        x = x.transpose(-1, -2)
    return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.lays = make_layouts(self.Y == self.X)
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

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].float().reshape(B, self.T, self.Y, self.X)
        ms = [fwd_t(m, l[0], l[1], l[2]) for l in self.lays]
        c = self.coords.expand(B, -1, -1, -1)
        h = torch.cat([obs] + ms + [c], 1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = None
        for l, out in zip(self.lays, self.outs):
            o = inv_t(out(f), l[0], l[1], l[2])
            place = o if place is None else place + o
        place = place.reshape(B, -1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
