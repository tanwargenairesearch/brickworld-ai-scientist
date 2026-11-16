import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
WH = 32
GH = 64
SCALE = 5.0


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def frame_list(Y, X):
    fr = []
    for tr in ([False, True] if Y == X else [False]):
        for fy in (False, True):
            for fx in (False, True):
                fr.append((tr, fy, fx))
    return fr


def apply_frame(t, fr):
    tr, fy, fx = fr
    if tr:
        t = t.transpose(2, 3)
    dims = []
    if fy:
        dims.append(2)
    if fx:
        dims.append(3)
    if len(dims) > 0:
        t = torch.flip(t, dims)
    return t


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(c, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, COL, 1)), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = init_layer(nn.Conv2d(c, SKIP, 1))

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        return f, torch.cat([f, self.skip(obs)], dim=1)


class WindowHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        K = max(max(a, b) for a, b in fps)
        k = 2 * K - 1
        self.T = spec.n_types
        self.frames = frame_list(spec.Y, spec.X)
        self.nf = len(self.frames)
        self.win = init_layer(nn.Conv2d(cin, WH, k, padding=K - 1))
        self.out = init_layer(nn.Conv2d(WH, self.nf * self.T, 1), 0.01)
        self.w = nn.Parameter(torch.zeros(2 * self.nf))
        self.dense = init_layer(nn.Linear(GH, spec.n_place), 0.01)

    def forward(self, h, g, tbias):
        B, _, Y, X = h.shape
        m = self.out(F.relu(self.win(h)))
        m = m.reshape(B, self.nf, self.T, Y, X) + tbias[:, None, :, None, None]
        outs = []
        for i, fr in enumerate(self.frames):
            mi = apply_frame(m[:, i], fr)
            outs.append(mi.reshape(B, -1))
            outs.append(mi.permute(0, 2, 3, 1).reshape(B, -1))
        L = torch.stack(outs, dim=0)
        wts = torch.softmax(SCALE * self.w, dim=0)
        return torch.einsum('f,fbn->bn', wts, L) + self.dense(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, GH)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GH, 1), 0.01)
        self.value_head = init_layer(nn.Linear(GH, 1), 1.0)
        self.type_bias = init_layer(nn.Linear(GH, spec.n_types), 0.01)
        self.head = WindowHead(spec, WIDTH + SKIP)

    def forward(self, obs, mask):
        f, h = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(h, g, self.type_bias(g))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
