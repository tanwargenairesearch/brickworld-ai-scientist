import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "lr": 0.0005, "ent_coef": 0.01}

STEM = 64
COL = 48
WIDTH = 40
FLATC = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        na = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH, FLATC, 1)
        self.pi = nn.Linear(FLATC * Y * X, na)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.mask_in = nn.Linear(na, 64)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.glob2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        logits = self.pi(F.relu(self.red(f)).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        g = self.glob2(g + self.mask_in(mask.float()))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
