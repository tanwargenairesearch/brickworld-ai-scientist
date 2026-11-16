import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.98, "gae_lambda": 0.95, "lr": 0.0015, "ent_coef": 0.01}

W = 32
RED = 16
HID = 256


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_act = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.b1 = Block(W)
        self.glob1 = nn.Linear(2 * W, W)
        self.b2 = Block(W)
        self.red = nn.Conv2d(W, RED, 1)
        self.fc = nn.Linear(RED * self.Y * self.X, HID)
        self.mfc = nn.Linear(self.n_act, HID)
        self.pi = nn.Linear(HID, self.n_act)
        self.value = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        h = self.stem(obs)
        h = self.b1(h)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        h = F.relu(h + self.glob1(pooled)[:, :, None, None])
        h = self.b2(h)
        z = F.relu(self.red(h)).flatten(1)
        g = F.relu(self.fc(z) + self.mfc(mask.to(obs.dtype)))
        return self.pi(g), self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
