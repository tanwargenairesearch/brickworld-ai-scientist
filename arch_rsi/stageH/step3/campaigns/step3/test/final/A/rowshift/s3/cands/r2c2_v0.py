import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
GLOB = 64
RED = 12
DH = 512
MH = 64


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = int(spec.in_channels)
        self.stem = nn.Sequential(nn.Conv2d(c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        h = torch.cat([f, self.skip(obs)], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return h, g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        na = int(spec.n_actions)
        self.trunk = Trunk(spec)
        self.red = nn.Conv2d(WIDTH + SKIP, RED, 1)
        self.mproj = nn.Linear(na, MH)
        self.d1 = nn.Linear(RED * Y * X + GLOB + MH, DH)
        self.d2 = nn.Linear(DH, na)
        self.value_head = nn.Linear(GLOB, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        ortho(self.d2, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        z = F.relu(self.red(h)).flatten(1)
        mk = F.relu(self.mproj(mask.float()))
        hh = F.relu(self.d1(torch.cat([z, g, mk], dim=1)))
        logits = self.d2(hh)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
