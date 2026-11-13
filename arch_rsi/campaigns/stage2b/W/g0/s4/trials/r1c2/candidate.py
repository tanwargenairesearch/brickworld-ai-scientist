"""Residual conv body (width 48) with a footprint-aware placement head: per brick type a conv whose kernel equals
the footprint (anchored top-left), plus a min-pool over the footprint of a learned per-cell score."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HEAD = 32
GLOB = 64


class FootprintHead(nn.Module):
    def __init__(self, c_in, footprints):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c_in, 1, (dy, dx)) for dy, dx in self.fps])
        self.cell = nn.Conv2d(c_in, len(self.fps), 1)

    def forward(self, h):
        cell = self.cell(h)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            a = self.convs[t](hp)
            c = F.pad(cell[:, t:t + 1], (0, dx - 1, 0, dy - 1))
            m = -F.max_pool2d(-c, (dy, dx), stride=1)
            outs.append(a + m)
        return torch.cat(outs, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head_in = nn.Conv2d(WIDTH, HEAD, 1)
        self.place = FootprintHead(HEAD, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        f = f + F.relu(self.c1(f))
        f = f + F.relu(self.c2(f))
        h = F.relu(self.head_in(f))
        place = self.place(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
