"""Per-cell MLP stem, one residual 3x3 context conv, and a top-left-anchored 2x2 footprint head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
FOOT = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.ctx = nn.Conv2d(W, W, 3, padding=1)
        self.foot = nn.Conv2d(W, FOOT, 2)
        self.place_head = nn.Conv2d(FOOT, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        h = self.stem(obs)
        f = h + F.relu(self.ctx(h))
        p = F.relu(self.foot(F.pad(f, (0, 1, 0, 1))))
        place = self.place_head(p).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
