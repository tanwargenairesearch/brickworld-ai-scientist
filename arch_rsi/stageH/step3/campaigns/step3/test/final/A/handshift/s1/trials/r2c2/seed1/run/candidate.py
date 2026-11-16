import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
WHID = 32
H = 8
GH = 64
SQ2 = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = ortho(nn.Conv2d(cin, STEM, 1), SQ2)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), SQ2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), SQ2)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), SQ2)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), SQ2)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        return torch.cat([x, F.relu(self.skip(obs))], dim=1)


class DenseRemapHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.win = ortho(nn.Conv2d(cin, WHID, 2 * K - 1, padding=K - 1), SQ2)
        self.feat = ortho(nn.Conv2d(WHID, H, 1), 1.0)
        self.remap = ortho(nn.Linear(H * spec.Y * spec.X, spec.n_actions), 0.01)

    def forward(self, f):
        m = self.feat(F.relu(self.win(f)))
        return self.remap(m.flatten(1))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.head = DenseRemapHead(c, spec)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * c, GH), SQ2), nn.ReLU())
        self.value_head = ortho(nn.Linear(GH, 1), 1.0)
        self.dense = ortho(nn.Linear(GH, spec.n_actions), 0.01)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = self.head(f) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
