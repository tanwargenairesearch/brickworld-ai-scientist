import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gae_lambda": 0.95}

W = 48
BLOCKS = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ky = max(int(fp[0]) for fp in spec.footprints)
        self.kx = max(int(fp[1]) for fp in spec.footprints)
        self.stem = nn.Conv2d(spec.in_channels, W, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(BLOCKS)])
        self.place_head = nn.Conv2d(W, spec.n_types, (self.ky, self.kx))
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for conv in self.blocks:
            f = f + F.relu(conv(f))
        fp = F.pad(f, (0, self.kx - 1, 0, self.ky - 1))
        place = self.place_head(fp).flatten(1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
