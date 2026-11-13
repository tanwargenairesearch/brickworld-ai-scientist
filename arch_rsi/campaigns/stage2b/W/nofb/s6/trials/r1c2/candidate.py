import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DILATIONS = (1, 2)
RECIPE = {"ppo_epochs": 2, "gamma": 0.98, "gae_lambda": 0.95}


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.glob = nn.Linear(2 * c, c)

    def forward(self, f):
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.conv(f) + g[:, :, None, None])
        return f + h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([Block(WIDTH, d) for d in DILATIONS])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for b in self.blocks:
            f = b(f)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
