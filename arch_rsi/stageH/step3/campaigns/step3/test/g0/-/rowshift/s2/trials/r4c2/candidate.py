import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.003, "gamma": 0.99, "gae_lambda": 0.95}

W = 32
HH = 32


class Res(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=d, dilation=d)
        self.na = nn.GroupNorm(8, c)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.nb = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.relu(self.na(self.a(x)))
        return F.relu(x + self.nb(self.b(h)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.blocks = nn.Sequential(Res(W, 1), Res(W, 2), Res(W, 1))
        self.h1 = nn.Conv2d(W, HH, 3, padding=1)
        self.h2 = nn.Conv2d(HH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = self.blocks(self.stem(obs))
        h = F.relu(self.h1(x))
        place = self.h2(h).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
