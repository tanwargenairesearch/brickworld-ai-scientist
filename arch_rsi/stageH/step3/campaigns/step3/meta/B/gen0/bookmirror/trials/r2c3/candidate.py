import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0015, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
FC = 16
HIDDEN = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.comp = nn.Conv2d(WIDTH, FC, 1)
        self.fc = nn.Linear(FC * Y * X + 2 * WIDTH, HIDDEN)
        self.pi = nn.Linear(HIDDEN, int(spec.n_actions))
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        self.vh = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        c = F.relu(self.comp(f)).flatten(1)
        gp = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        h = F.relu(self.fc(torch.cat([c, gp], dim=1)))
        logits = self.pi(h)
        value = self.value_head(self.vh(h)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
