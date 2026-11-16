import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
PROJ = 32
SQH = 48
GHID = 64
MIX_SCALE = 5.0


def apply_frame(x, fr):
    tr, fy, fx = fr
    if tr:
        x = x.transpose(-1, -2)
    if fy:
        x = x.flip(-2)
    if fx:
        x = x.flip(-1)
    return x


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
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
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GHID), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, F.relu(self.skip(obs))], dim=1)
        return h, g


class SquareFrameHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.T = int(spec.n_types)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.K = max(max(a, b) for a, b in fps)
        if Y == X:
            self.frames = [(False, False, False), (True, False, False)]
        else:
            self.frames = [(False, False, False), (False, True, True)]
        self.proj = nn.Conv2d(cin, PROJ, 1)
        self.sq = nn.ModuleList([nn.Conv2d(PROJ, SQH, self.K) for _ in self.frames])
        self.outs = nn.ModuleList([nn.Conv2d(SQH, self.T, 1) for _ in self.frames])
        self.tmix = nn.Parameter(torch.zeros(self.T, self.T))

    def forward(self, h):
        p = F.relu(self.proj(h))
        res = []
        for fr, sq, out in zip(self.frames, self.sq, self.outs):
            q = apply_frame(p, fr)
            q = F.pad(q, (0, self.K - 1, 0, self.K - 1))
            z = F.relu(sq(q))
            o = out(z)
            o = o + torch.einsum('tu,buyx->btyx', self.tmix, o)
            res.append(o.flatten(1))
        return torch.stack(res, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.YX = int(spec.Y) * int(spec.X)
        self.trunk = Trunk(spec)
        self.head = SquareFrameHead(spec, WIDTH + SKIP)
        self.alpha = nn.Parameter(torch.zeros(len(self.head.frames)))
        self.dense = nn.Linear(GHID, int(spec.n_place))
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.type_bias = nn.Linear(GHID, self.T)
        init_layers(self)
        for m in self.head.outs:
            init_out(m, 0.01)
        init_out(self.dense, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        L = self.head(h)
        w = torch.softmax(MIX_SCALE * self.alpha, 0)
        place = torch.einsum('bgn,g->bn', L, w)
        tb = self.type_bias(g).unsqueeze(2).expand(-1, self.T, self.YX).reshape(obs.shape[0], -1)
        place = place + tb + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
