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
HS = 40
SCALE = 5.0


def apply_frame(t, fr):
    tr, fy, fx = fr
    if tr:
        t = t.transpose(-1, -2)
    dims = []
    if fy:
        dims.append(-2)
    if fx:
        dims.append(-1)
    if len(dims) > 0:
        t = torch.flip(t, dims)
    return t


def init_all(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        feat = torch.cat([f, self.skip(obs)], 1)
        return feat, g


class SquareHead(nn.Module):
    def __init__(self, cin, footprints, Y, X, n_place):
        super().__init__()
        self.T = len(footprints)
        self.K = max(max(int(a), int(b)) for a, b in footprints)
        if Y == X:
            self.frames = [(False, False, False), (True, False, False)]
        else:
            self.frames = [(False, False, False), (False, True, True)]
        self.convs = nn.ModuleList([nn.Conv2d(cin, HS, self.K) for fr in self.frames])
        self.outs = nn.ModuleList([nn.Conv2d(HS, self.T, 1) for fr in self.frames])
        self.tmix = nn.Parameter(torch.zeros(self.T, self.T))
        self.alpha = nn.Parameter(torch.zeros(len(self.frames)))
        self.dense = nn.Linear(GH, n_place)

    def forward(self, feat, g):
        K = self.K
        maps = []
        for fr, conv, out in zip(self.frames, self.convs, self.outs):
            q = apply_frame(feat, fr)
            q = F.pad(q, (0, K - 1, 0, K - 1))
            m = out(F.relu(conv(q)))
            m = m + torch.einsum('st,btyx->bsyx', self.tmix, m)
            maps.append(m.flatten(1))
        S = torch.stack(maps, 0)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        return torch.einsum('k,kbn->bn', w, S) + self.dense(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.head = SquareHead(WIDTH + SKIP, spec.footprints, int(spec.Y), int(spec.X), int(spec.n_place))
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        init_all(self)
        for m in self.head.outs:
            init_out(m, 0.01)
        init_out(self.head.dense, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        place = self.head(feat, g)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
