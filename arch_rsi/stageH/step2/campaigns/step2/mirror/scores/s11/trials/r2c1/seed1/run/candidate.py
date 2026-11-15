import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 2, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
RED = 16
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cells = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH, RED, 1)
        self.fc = nn.Linear(RED * cells + 2 * WIDTH, HID)
        self.pi = nn.Linear(HID, spec.n_actions)
        self.vh = nn.Linear(HID, 64)
        self.v = nn.Linear(64, 1)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.red(f)).flatten(1)
        g = torch.cat([r, f.mean((2, 3)), f.amax((2, 3))], dim=1)
        h = F.relu(self.fc(g))
        logits = self.pi(h)
        value = self.v(F.relu(self.vh(h))).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
