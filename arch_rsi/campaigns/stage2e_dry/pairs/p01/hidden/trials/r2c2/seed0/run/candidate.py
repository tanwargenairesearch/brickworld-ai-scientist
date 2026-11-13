"""Residual conv body with per-footprint max+mean pooled placement heads (pooling after learned layers)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
N_RES = 2
HEAD_HID = 32

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 4, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.inp = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.fps = [(int(dy), int(dx)) for (dy, dx) in spec.footprints]
        heads = []
        for _ in self.fps:
            heads.append(nn.Sequential(
                nn.Conv2d(2 * WIDTH, HEAD_HID, 1),
                nn.ReLU(),
                nn.Conv2d(HEAD_HID, 1, 1),
            ))
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.inp(obs))
        for conv in self.blocks:
            f = F.relu(f + conv(f))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
