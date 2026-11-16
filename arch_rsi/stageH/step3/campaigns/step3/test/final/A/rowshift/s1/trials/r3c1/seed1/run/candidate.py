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
PRE = 24
SCALE = 5.0
RLO = 16


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


def roll_mix(v, wlo, whi, rlo):
    r = 0
    for j in range(wlo.shape[0]):
        r = r + wlo[j] * torch.roll(v, j, 1)
    out = 0
    for i in range(whi.shape[0]):
        out = out + whi[i] * torch.roll(r, i * rlo, 1)
    return out


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


class RollHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(fps)
        K = max(max(a, b) for a, b in fps)
        self.frames = frame_list(spec.Y, spec.X)
        nf = len(self.frames)
        self.pre = ortho(nn.Conv2d(cin, PRE, 1), math.sqrt(2.0))
        self.win = ortho(nn.Conv2d(PRE, nf * self.T, 2 * K - 1, padding=K - 1), 0.01)
        n = spec.n_actions
        self.rlo = RLO
        nhi = (n + RLO - 1) // RLO
        self.mix_f = nn.Parameter(torch.zeros(nf))
        self.mix_o = nn.Parameter(torch.zeros(2))
        self.mix_lo = nn.Parameter(torch.zeros(RLO))
        self.mix_hi = nn.Parameter(torch.zeros(nhi))
        self.mix_d = nn.Parameter(torch.zeros(2))
        self.tbias = ortho(nn.Linear(GH, self.T), 0.01)
        self.done = ortho(nn.Linear(GH, 1), 0.01)
        self.dense = ortho(nn.Linear(GH, n), 0.01)

    def forward(self, feat, g):
        T = self.T
        M = self.win(F.relu(self.pre(feat)))
        wf = F.softmax(SCALE * self.mix_f, 0)
        L = 0
        for i, fr in enumerate(self.frames):
            L = L + wf[i] * apply_frame(M[:, i * T:(i + 1) * T], fr)
        L = L + self.tbias(g)[:, :, None, None]
        wo = F.softmax(SCALE * self.mix_o, 0)
        place = wo[0] * L.flatten(1) + wo[1] * L.permute(0, 2, 3, 1).flatten(1)
        d = self.done(g)
        wlo = F.softmax(SCALE * self.mix_lo, 0)
        whi = F.softmax(SCALE * self.mix_hi, 0)
        wd = F.softmax(SCALE * self.mix_d, 0)
        full = roll_mix(torch.cat([place, d], 1), wlo, whi, self.rlo)
        part = torch.cat([roll_mix(place, wlo, whi, self.rlo), d], 1)
        return wd[0] * full + wd[1] * part + self.dense(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.head = RollHead(spec, self.trunk.out_ch)
        self.value_head = ortho(nn.Linear(GH, 1), 1.0)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        logits = self.head(feat, g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
