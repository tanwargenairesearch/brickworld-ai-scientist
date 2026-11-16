import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.003}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GH = 64
PRE = 24
SCALE = 5.0


def frame_list(Y, X):
    trs = [False, True] if Y == X else [False]
    return [(tr, fy, fx) for tr in trs for fy in (False, True) for fx in (False, True)]


def apply_frame(x, fr):
    if fr[0]:
        x = x.transpose(2, 3)
    if fr[1]:
        x = torch.flip(x, [2])
    if fr[2]:
        x = torch.flip(x, [3])
    return x


def ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
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


class WindowHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(fps)
        K = max(max(a, b) for a, b in fps)
        self.frames = frame_list(spec.Y, spec.X)
        nf = len(self.frames)
        self.pre = ortho(nn.Conv2d(cin, PRE, 1), math.sqrt(2.0))
        self.win = ortho(nn.Conv2d(PRE, 2 * nf * self.T, 2 * K - 1, padding=K - 1), 0.01)
        self.mix = nn.Parameter(torch.zeros(2 * nf))
        self.dense = ortho(nn.Linear(GH, spec.n_place), 0.01)
        self.tbias = ortho(nn.Linear(GH, self.T), 0.01)

    def forward(self, feat, g):
        M = self.win(F.relu(self.pre(feat)))
        tb = self.tbias(g)[:, :, None, None]
        T = self.T
        cands = []
        for i, fr in enumerate(self.frames):
            a = apply_frame(M[:, (2 * i) * T:(2 * i + 1) * T], fr) + tb
            b = apply_frame(M[:, (2 * i + 1) * T:(2 * i + 2) * T], fr) + tb
            cands.append(a.flatten(1))
            cands.append(b.flatten(2).transpose(1, 2).flatten(1))
        C = torch.stack(cands, 1)
        w = F.softmax(SCALE * self.mix, 0)
        return (C * w[None, :, None]).sum(1) + self.dense(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.head = WindowHead(spec, self.trunk.out_ch)
        self.done_head = ortho(nn.Linear(GH, 1), 0.01)
        self.value_head = ortho(nn.Linear(GH, 1), 1.0)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        place = self.head(feat, g)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
