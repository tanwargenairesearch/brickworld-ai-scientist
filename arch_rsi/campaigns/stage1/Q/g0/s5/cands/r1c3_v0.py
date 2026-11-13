import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
BLOCKS = 2


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.glob = nn.Linear(2 * w, w)

    def forward(self, x):
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        h = self.conv(x) + self.glob(pooled)[:, :, None, None]
        return x + F.relu(h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(BLOCKS)])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.globh = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for b in self.blocks:
            f = b(f)
        place = self.place_head(f).flatten(1)
        g = self.globh(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
