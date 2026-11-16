import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64, "lr": 0.0007, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

CELL_H = 32
CELL_O = 12
HID = 512


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.cell = nn.Sequential(
            nn.Conv2d(spec.in_channels, CELL_H, 1), nn.ReLU(),
            nn.Conv2d(CELL_H, CELL_O, 1), nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(CELL_O * Y * X, HID), nn.ReLU(),
            nn.Linear(HID, HID), nn.ReLU())
        self.pi = nn.Linear(HID, spec.n_actions)
        self.v = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        c = self.cell(obs).flatten(1)
        h = self.mlp(c)
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
