import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gamma": 0.97}

RED = 32
H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.red = nn.Conv2d(spec.in_channels, RED, 1)
        self.fc1 = nn.Linear(RED * Y * X, H)
        self.fc2 = nn.Linear(H, H)
        self.pi = nn.Linear(H, spec.n_actions)
        self.v = nn.Linear(H, 1)

    def forward(self, obs, mask):
        h = F.relu(self.red(obs)).flatten(1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
