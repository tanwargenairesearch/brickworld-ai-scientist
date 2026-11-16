import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.01}

STEM = 64
COL = 48
WIDTH = 40
NARROW = 8
HIDDEN = 192


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cells = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.GroupNorm(1, STEM), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.GroupNorm(1, COL), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(1, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(1, WIDTH)
        self.narrow = nn.Conv2d(WIDTH, NARROW, 1)
        self.fc = nn.Sequential(nn.Linear(NARROW * cells + 2 * WIDTH, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU())
        self.pi = nn.Linear(HIDDEN, spec.n_actions)
        self.value_head = nn.Linear(HIDDEN, 1)
        nn.init.zeros_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        n = F.relu(self.narrow(f)).flatten(1)
        h = self.fc(torch.cat([n, f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return self.pi(h), self.value_head(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
