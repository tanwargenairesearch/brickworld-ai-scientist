"""Per-cell stem, residual 3x3 context with global broadcast, and per-type footprint max/min pooling head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
PW = 32
HID = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        self.bcast = nn.Linear(2 * W, W)
        self.proj = nn.Conv2d(W, PW, 1)
        self.mix = nn.Conv2d(2 * PW + W, HID, 1)
        self.type_bias = nn.Parameter(torch.zeros(spec.n_types, HID, 1, 1))
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(spec.n_types)])
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        h = self.stem(obs)
        f = h + F.relu(self.ctx(h))
        g0 = F.relu(self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        f = f + g0[:, :, None, None]
        p = self.proj(f)
        place = []
        for t, (dy, dx) in enumerate(self.footprints):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            z = F.relu(self.mix(torch.cat([mx, mn, f], dim=1)) + self.type_bias[t])
            place.append(self.outs[t](z).flatten(1))
        place = torch.cat(place, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
