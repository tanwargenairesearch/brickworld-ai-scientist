import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.003, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
U = 24
H = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = len(spec.footprints)
        T = self.T
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.u = nn.Conv2d(COL + WIDTH, U, 1)
        self.h1 = nn.Conv2d(U, T * H, 3, padding=1)
        self.h2 = nn.Conv2d(T * H, T * H, 1, groups=T)
        self.h3 = nn.Conv2d(T * H, T, 1, groups=T)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + U, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        u = F.relu(self.u(torch.cat([s, f], 1)))
        h = F.relu(self.h1(u))
        h = F.relu(self.h2(h))
        place = self.h3(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), u.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
