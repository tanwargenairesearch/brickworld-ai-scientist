import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 40
NBLK = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLK)])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for conv in self.blocks:
            x = x + F.relu(conv(x))
        place = self.place_head(x).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
