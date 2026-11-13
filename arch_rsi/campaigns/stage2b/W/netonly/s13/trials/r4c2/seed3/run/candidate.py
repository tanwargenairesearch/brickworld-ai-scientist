import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 36
NBLOCKS = 3
CW = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLOCKS)])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.c1 = nn.Conv2d(spec.in_channels, CW, 3, padding=1)
        self.c2 = nn.Conv2d(CW, CW, 3, padding=1)
        self.cglob = nn.Sequential(nn.Linear(2 * CW, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for conv in self.blocks:
            f = f + F.relu(conv(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        c = F.relu(self.c1(obs))
        c = c + F.relu(self.c2(c))
        cg = self.cglob(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1))
        return logits, self.value_head(cg).squeeze(1)


def build_policy(spec):
    return Net(spec)
