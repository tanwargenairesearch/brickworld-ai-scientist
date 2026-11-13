"""Conv stem plus one residual block whose hidden layer receives a broadcast global context vector."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)
        self.g = nn.Linear(2 * w, w)

    def forward(self, f):
        g = self.g(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.c1(f) + g[:, :, None, None])
        return F.relu(f + self.c2(h))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.block = Block(WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        f = self.block(f)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
