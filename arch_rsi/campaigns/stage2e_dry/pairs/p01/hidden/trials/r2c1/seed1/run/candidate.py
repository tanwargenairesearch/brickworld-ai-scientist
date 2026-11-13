"""r1c2 network (residual conv body width 48, footprint-shaped heads) with a 2-epoch recipe for more samples."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
N_RES = 2
HEAD_HID = 16

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.inp = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
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
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.inp(obs))
        for conv in self.blocks:
            f = F.relu(f + conv(f))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(head(p))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
