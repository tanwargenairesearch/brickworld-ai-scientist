"""Residual conv body with a mid-network global-context broadcast and a 2x2 footprint placement head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

W = 34


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, h):
        return F.relu(h + self.c2(F.relu(self.c1(h))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.block1 = Block(W)
        self.bcast = nn.Linear(2 * W, W)
        self.block2 = Block(W)
        self.place_head = nn.Conv2d(W, spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = self.block1(h)
        g0 = self.bcast(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h = F.relu(h + g0[:, :, None, None])
        h = self.block2(h)
        place = self.place_head(F.pad(h, (0, 1, 0, 1))).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
