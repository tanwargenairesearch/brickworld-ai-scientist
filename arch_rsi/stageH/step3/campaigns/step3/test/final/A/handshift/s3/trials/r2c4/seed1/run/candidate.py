import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
GLOB = 64
KF = 16
ADDR_SCALE = 3.0
R2 = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, spec, width, skip):
        super().__init__()
        c = spec.in_channels
        self.stem1 = ortho(nn.Conv2d(c, STEM, 1), R2)
        self.stem2 = ortho(nn.Conv2d(STEM, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, width, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(4, width)
        self.c2 = ortho(nn.Conv2d(width, width, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(4, width)
        self.skip = ortho(nn.Conv2d(c, skip, 1), 1.0)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * width, GLOB), R2), nn.ReLU())
        self.out_ch = width + skip

    def forward(self, obs):
        s = F.relu(self.stem1(obs))
        s = F.relu(self.stem2(s))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, self.skip(obs)], dim=1)
        return h, g


class PointerHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        YX = int(spec.Y) * int(spec.X)
        A = int(spec.n_actions)
        self.proj = ortho(nn.Conv2d(cin, KF, 1), R2)
        self.addr = nn.Parameter(torch.zeros(A, YX))
        emb = torch.empty(A, KF)
        nn.init.normal_(emb, 0.0, 0.05)
        self.emb = nn.Parameter(emb)
        self.bias = nn.Parameter(torch.zeros(A))
        self.dense = ortho(nn.Linear(GLOB, A), 0.01)

    def forward(self, h, g):
        u = F.relu(self.proj(h)).flatten(2)
        w = torch.softmax(ADDR_SCALE * self.addr, dim=-1)
        v = torch.einsum('ap,bkp->bak', w, u)
        logits = (v * self.emb.unsqueeze(0)).sum(-1)
        return logits + self.bias + self.dense(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec, WIDTH, SKIP)
        self.head = PointerHead(self.trunk.out_ch, spec)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        logits = self.head(h, g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
