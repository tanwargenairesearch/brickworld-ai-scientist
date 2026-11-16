import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 36
SKIP = 16
LOC = 16


def ortho_init(net):
    for mod in net.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(mod.weight, math.sqrt(2.0))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def set_gain(mod, gain):
    nn.init.orthogonal_(mod.weight, gain)
    nn.init.zeros_(mod.bias)


class Trunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c, 64, 1), nn.ReLU(), nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return torch.cat([f, k], dim=1), g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_actions = int(spec.n_actions)
        self.trunk = Trunk(int(spec.in_channels))
        self.loc = nn.Conv2d(WIDTH + SKIP, LOC, 3, padding=1)
        self.readout = nn.Linear(LOC * self.Y * self.X, self.n_actions)
        self.gdense = nn.Linear(64, self.n_actions)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self)
        set_gain(self.readout, 0.01)
        set_gain(self.gdense, 0.01)
        set_gain(self.value_head, 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        l = F.relu(self.loc(h)).flatten(1)
        logits = self.readout(l) + self.gdense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
