"""Residual conv body with a dilated block and a global-context broadcast, footprint-shaped heads, 2 PPO epochs."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HEAD_HID = 16

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.inp = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.b1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.b2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.fps = [(int(dy), int(dx)) for (dy, dx) in spec.footprints]
        heads = []
        for (dy, dx) in self.fps:
            heads.append(nn.Sequential(
                nn.Conv2d(WIDTH, HEAD_HID, (dy, dx)),
                nn.ReLU(),
                nn.Conv2d(HEAD_HID, 1, 1),
            ))
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.bcast = nn.Linear(WIDTH, WIDTH)
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.inp(obs))
        f = F.relu(f + self.b1(f))
        f = F.relu(f + self.b2(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(f + self.bcast(g)[:, :, None, None])
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            outs.append(head(p))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
