import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
CELL = 32
H1 = 320
H2 = 256


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        g = math.sqrt(2.0)
        self.s1 = ortho(nn.Conv2d(spec.in_channels, STEM, 1), g)
        self.s2 = ortho(nn.Conv2d(STEM, CELL, 1), g)
        self.fc1 = ortho(nn.Linear(CELL * spec.Y * spec.X, H1), g)
        self.fc2 = ortho(nn.Linear(H1, H2), g)
        self.pi = ortho(nn.Linear(H2, spec.n_actions), 0.01)
        self.value = ortho(nn.Linear(H2, 1), 1.0)

    def forward(self, obs, mask):
        h = F.relu(self.s2(F.relu(self.s1(obs)))).flatten(1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.pi(h), self.value(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
