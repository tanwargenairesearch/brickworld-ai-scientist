"""Residual 3-conv body (width 46, compute <= starting net), 1x1 placement head, 1 PPO epoch."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 46
N_RES = 2

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.inp = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.inp(obs))
        for conv in self.blocks:
            f = F.relu(f + conv(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
