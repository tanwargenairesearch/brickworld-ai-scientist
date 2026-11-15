import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

W = 32
RED = 8
HID = 384


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.blocks = nn.Sequential(Block(W), Block(W))
        self.red = nn.Conv2d(W, RED, 1)
        n_in = RED * int(spec.Y) * int(spec.X)
        self.fc = nn.Sequential(nn.Linear(n_in, HID), nn.ReLU())
        self.pi = nn.Linear(HID, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        f = self.blocks(self.stem(obs))
        z = F.relu(self.red(f)).flatten(1)
        h = self.fc(z)
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
