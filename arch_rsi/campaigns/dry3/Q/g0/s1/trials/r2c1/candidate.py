import torch
import torch.nn as nn
import torch.nn.functional as F

COL = 64
WIDTH = 40
GLOB = 64
NBLOCKS = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, COL, 1), nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.films = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(NBLOCKS)])
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLOCKS)])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        h = self.col(obs)
        f = F.relu(self.conv1(h))
        for film, conv in zip(self.films, self.convs):
            g = film(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
            f = f + F.relu(conv(f + g[:, :, None, None]))
        place = self.place_head(F.pad(f, (0, 1, 0, 1))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
