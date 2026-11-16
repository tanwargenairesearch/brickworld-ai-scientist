import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
WHID = 32
GLOB = 64
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def frame_maps(Y, X):
    maps = [lambda y, x: (y, x),
            lambda y, x: (y, X - 1 - x),
            lambda y, x: (Y - 1 - y, x),
            lambda y, x: (Y - 1 - y, X - 1 - x)]
    if Y == X:
        maps = maps + [lambda y, x: (x, y),
                       lambda y, x: (x, X - 1 - y),
                       lambda y, x: (Y - 1 - x, y),
                       lambda y, x: (Y - 1 - x, X - 1 - y)]
    return maps


def build_index(Y, X, T, maps):
    YX = Y * X
    P = T * YX
    rows = []
    for g, m in enumerate(maps):
        for order in range(2):
            row = []
            for a in range(P):
                if order == 0:
                    t = a // YX
                    c = a % YX
                else:
                    t = a % T
                    c = a // T
                y, x = m(c // X, c % X)
                row.append(g * P + t * YX + y * X + x)
            rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
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
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(fps)
        K = max(max(a, b) for a, b in fps)
        maps = frame_maps(Y, X)
        self.nf = len(maps)
        self.trunk = Trunk(spec)
        self.win = nn.Conv2d(WIDTH + SKIP, WHID, 2 * K - 1, padding=K - 1)
        self.wout = nn.Conv2d(WHID, self.nf * T, 1)
        self.dense = nn.Linear(GLOB, int(spec.n_actions))
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        self.frame_logit = nn.Parameter(torch.zeros(2 * self.nf))
        self.register_buffer("idx", build_index(Y, X, T, maps))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        ortho(self.wout, 0.01)
        ortho(self.dense, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        m = self.wout(F.relu(self.win(h)))
        flat = m.flatten(1)
        per = flat[:, self.idx]
        w = torch.softmax(SCALE * self.frame_logit, 0)
        place = torch.einsum('bfp,f->bp', per, w)
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
