"""Residual narrow-deep conv body + footprint-pooled per-type placement heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
HID = 32
DILATIONS = [1, 1, 2, 1]


def fp_pool(f, dy, dx):
    p = F.pad(f, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    return mx, mn


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 1)
        self.convs = nn.ModuleList([
            nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILATIONS])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = self.stem(obs)
        for conv in self.convs:
            h = h + conv(F.relu(h))
        f = F.relu(h)
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            mx, mn = fp_pool(f, dy, dx)
            outs.append(head(torch.cat([f, mx, mn], 1)).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = torch.cat(outs + [self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
