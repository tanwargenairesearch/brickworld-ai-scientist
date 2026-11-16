import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
GP = 16
GH = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_cells = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.gproj = nn.Conv2d(WIDTH, GP, 1)
        self.fc1 = nn.Linear(GP * n_cells + 64, GH)
        self.fc2 = nn.Linear(GH, int(spec.n_actions))
        nn.init.orthogonal_(self.fc2.weight, 0.01)
        nn.init.zeros_(self.fc2.bias)
        self.value_head = nn.Linear(GH + 64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        gf = F.relu(self.gproj(f)).flatten(1)
        h = F.relu(self.fc1(torch.cat([gf, g], dim=1)))
        logits = self.fc2(h)
        value = self.value_head(torch.cat([h, g], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
