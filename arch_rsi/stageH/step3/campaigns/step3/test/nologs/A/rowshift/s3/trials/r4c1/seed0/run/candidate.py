import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
FLAT = 16
GLOB = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH, FLAT, 1)
        n = FLAT * int(spec.Y) * int(spec.X)
        self.pol = nn.Linear(n, int(spec.n_actions))
        nn.init.orthogonal_(self.pol.weight, gain=0.01)
        nn.init.zeros_(self.pol.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.value_head = nn.Linear(GLOB, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.red(f))
        logits = self.pol(r.flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
