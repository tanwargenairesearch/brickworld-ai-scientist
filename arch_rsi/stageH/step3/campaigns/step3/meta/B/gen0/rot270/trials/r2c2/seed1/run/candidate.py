import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.95, "gae_lambda": 0.9}

STEM = 64
W = 32
HH = 64
G = 64


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.b1 = Res(W)
        self.b2 = Res(W)
        self.h1 = nn.Conv2d(W, HH, 1)
        self.h2 = nn.Conv2d(HH, self.n_types, 3, padding=1)
        nn.init.orthogonal_(self.h2.weight.view(self.n_types, -1), 0.01)
        nn.init.zeros_(self.h2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        f = self.b2(self.b1(s))
        place = self.h2(F.relu(self.h1(f))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
