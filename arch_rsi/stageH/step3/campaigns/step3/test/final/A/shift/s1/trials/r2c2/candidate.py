import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
MID = 32
KMAP = 16
GLOB = 64
R2 = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = ortho(nn.Conv2d(cin, STEM, 1), R2)
        self.n0 = nn.GroupNorm(8, STEM)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), R2)
        self.nk = nn.GroupNorm(4, SKIP)

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.n0(self.s1(obs)))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.nk(self.skip(obs)))
        return torch.cat([x, k], 1), x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.pre = ortho(nn.Conv2d(WIDTH + SKIP, MID, 3, padding=1), R2)
        self.maps = ortho(nn.Conv2d(MID, KMAP, 1), 1.0)
        self.remap = ortho(nn.Linear(KMAP * spec.Y * spec.X, spec.n_actions), 0.01)
        self.glob_in = ortho(nn.Linear(2 * WIDTH, GLOB), R2)
        self.mask_in = ortho(nn.Linear(spec.n_actions, GLOB), 0.5)
        self.dense = ortho(nn.Linear(GLOB, spec.n_actions), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        f, x = self.trunk(obs)
        mf = mask.to(obs.dtype)
        g = F.relu(self.glob_in(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)) + self.mask_in(mf))
        mp = self.maps(F.relu(self.pre(f)))
        logits = self.remap(mp.flatten(1)) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
