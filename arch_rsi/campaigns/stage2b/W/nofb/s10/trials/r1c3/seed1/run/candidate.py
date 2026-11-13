"""Residual 3x3 conv net (width 48) with mask channels as input and a global-context broadcast between blocks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.n_types = spec.n_types
        self.n_place = spec.n_place
        self.Y = spec.Y
        self.X = spec.X
        self.stem = nn.Conv2d(spec.in_channels + spec.n_types, W, 3, padding=1)
        self.block1 = nn.Conv2d(W, W, 3, padding=1)
        self.bcast = nn.Linear(2 * W, W)
        self.block2 = nn.Conv2d(W, W, 3, padding=1)
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.n_types, self.Y, self.X)
        x = torch.cat([obs, m], dim=1)
        f = F.relu(self.stem(x))
        f = f + F.relu(self.block1(f))
        g0 = F.relu(self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        f = f + g0[:, :, None, None]
        f = f + F.relu(self.block2(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
