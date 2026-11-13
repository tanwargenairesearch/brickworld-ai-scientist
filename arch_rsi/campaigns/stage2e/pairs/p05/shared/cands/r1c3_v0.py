import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.block1 = nn.Conv2d(W, W, 3, padding=1)
        self.ctx = nn.Linear(2 * W, W)
        self.block2 = nn.Conv2d(W, W, 3, padding=1)
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = h + F.relu(self.block1(h))
        c = self.ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h = F.relu(h + c[:, :, None, None])
        h = h + F.relu(self.block2(h))
        place = self.place_head(h).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
