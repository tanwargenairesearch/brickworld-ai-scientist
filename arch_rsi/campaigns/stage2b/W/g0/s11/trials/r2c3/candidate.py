"""Residual conv body (width 48, 4 convs, coord channels) with footprint-pooled placement head, 3 PPO epochs."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
N_RES = 3

RECIPE = {"ppo_epochs": 3}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels + 2, WIDTH, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.pre = nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 1), nn.ReLU())
        self.heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        f = torch.relu(self.stem(torch.cat([obs, ys, xs], dim=1)))
        for conv in self.convs:
            f = torch.relu(f + conv(f))
        h = self.pre(f)
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            outs.append(head(torch.cat([h, mx, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
