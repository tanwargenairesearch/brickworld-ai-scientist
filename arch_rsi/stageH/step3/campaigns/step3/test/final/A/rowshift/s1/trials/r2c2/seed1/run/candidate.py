import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GH = 64
KR = 16


def ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Trunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        r2 = math.sqrt(2.0)
        self.s1 = ortho(nn.Conv2d(c, STEM, 1), r2)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), r2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), r2)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), r2)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), 1.0)
        self.glob = ortho(nn.Linear(2 * WIDTH, GH), r2)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        feat = torch.cat([f, self.skip(obs)], 1)
        return feat, g


class ReadoutHead(nn.Module):
    def __init__(self, spec, cin, kr, gain):
        super().__init__()
        n = spec.Y * spec.X
        self.red = ortho(nn.Conv2d(cin, kr, 1), math.sqrt(2.0))
        w = torch.empty(spec.n_actions, kr * n)
        nn.init.orthogonal_(w, gain)
        self.w = nn.Parameter(w)

    def forward(self, feat):
        r = F.relu(self.red(feat)).flatten(1)
        return r @ self.w.t()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.read = ReadoutHead(spec, self.trunk.out_ch, KR, 0.01)
        self.dense = ortho(nn.Linear(GH, spec.n_actions), 0.01)
        self.value_head = ortho(nn.Linear(GH, 1), 1.0)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        logits = self.read(feat) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
