import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 47


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.block1 = Block(WIDTH)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.block2 = Block(WIDTH)
        self.final_norm = nn.GroupNorm(1, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = self.block1(x)
        g0 = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        x = x + self.ctx(g0)[:, :, None, None]
        x = self.block2(x)
        f = F.relu(self.final_norm(x))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
