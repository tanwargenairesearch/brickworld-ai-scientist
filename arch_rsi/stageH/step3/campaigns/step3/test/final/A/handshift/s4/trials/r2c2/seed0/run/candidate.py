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
DPROJ = 8
GHID = 64
MIX_SCALE = 5.0


def make_frames(Y, X):
    frames = []
    trs = [False, True] if Y == X else [False]
    for tr in trs:
        for fy in (False, True):
            for fx in (False, True):
                frames.append((tr, fy, fx))
    return frames


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


class WindowFrameHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        K = max(max(a, b) for a, b in fps)
        self.frames = make_frames(self.Y, self.X)
        self.nf = len(self.frames)
        self.proj = nn.Conv2d(cin, PROJ, 1)
        self.win = nn.Conv2d(PROJ, self.nf * self.T, 2 * K - 1, padding=K - 1)

    def forward(self, h):
        m = self.win(F.relu(self.proj(h)))
        B = m.shape[0]
        m = m.reshape(B, self.nf, self.T, self.Y, self.X)
        outs = []
        for i, fr in enumerate(self.frames):
            mi = apply_frame(m[:, i], fr)
            outs.append(mi.flatten(1))
            outs.append(mi.permute(0, 2, 3, 1).flatten(1))
        return torch.stack(outs, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.YX = int(spec.Y) * int(spec.X)
        self.trunk = Trunk(spec)
        self.head = WindowFrameHead(spec, WIDTH + SKIP)
        self.alpha = nn.Parameter(torch.zeros(2 * self.head.nf))
        self.dproj = nn.Conv2d(WIDTH + SKIP, DPROJ, 1)
        self.dense_sp = nn.Linear(DPROJ * self.YX, int(spec.n_actions))
        self.dense = nn.Linear(GHID, int(spec.n_place))
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.type_bias = nn.Linear(GHID, self.T)
        init_layers(self)
        init_out(self.head.win, 0.01)
        init_out(self.dense_sp, 0.01)
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
        z = F.relu(self.dproj(h)).flatten(1)
        logits = logits + self.dense_sp(z)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
