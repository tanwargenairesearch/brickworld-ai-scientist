import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 48
GLOB = 64


def make_layouts(square):
    lay = []
    for t in ([False, True] if square else [False]):
        for fy in [False, True]:
            for fx in [False, True]:
                lay.append((t, fy, fx))
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

    def place(self, f, outs):
        p = None
        for l, out in zip(self.lays, outs):
            o = inv_t(out(f), l[0], l[1], l[2])
            p = o if p is None else p + o
        return p.reshape(f.shape[0], -1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        ma = mask[:, :self.n_place].float().reshape(B, self.T, self.Y, self.X)
        mb = mask[:, 1:self.n_place + 1].float().reshape(B, self.T, self.Y, self.X)
        ms = [fwd_t(ma, l[0], l[1], l[2]) for l in self.lays]
        ms = ms + [fwd_t(mb, l[0], l[1], l[2]) for l in self.lays]
        c = self.coords.expand(B, -1, -1, -1)
        h = torch.cat([obs] + ms + [c], 1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        pa = self.place(f, self.outs_a)
        pb = self.place(f, self.outs_b)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([pa, self.done_a(g)], dim=1) + torch.cat([self.done_b(g), pb], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
