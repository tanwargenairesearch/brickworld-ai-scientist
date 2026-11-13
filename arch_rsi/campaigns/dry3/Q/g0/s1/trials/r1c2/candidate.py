import torch
import torch.nn as nn
import torch.nn.functional as F

COL = 64
WIDTH = 48
GLOB = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, COL, 1), nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.film = nn.Linear(2 * WIDTH, WIDTH)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        h = self.col(obs)
        f = F.relu(self.conv1(h))
        g1 = self.film(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(self.conv2(f + g1[:, :, None, None]))
        place = self.place_head(F.pad(f, (0, 1, 0, 1))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
