import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

GC = 2
COL = 48
WIDTH = 32
SKIP = 16
MID = 32
GLOB = 64
SCALE = 5.0
R2 = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def frame_list(Y, X):
    ts = [False, True] if Y == X else [False]
    out = []
    for t in ts:
        for fy in (False, True):
            for fx in (False, True):
                out.append((t, fy, fx))
    return out


def to_frame(z, fr):
    t, fy, fx = fr
    if t:
        z = z.transpose(-1, -2)
    dims = []
    if fy:
        dims.append(-2)
    if fx:
        dims.append(-1)
    if len(dims) > 0:
        z = torch.flip(z, dims)
    return z


class RemapTrunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        C = int(spec.in_channels)
        self.Z = Z
        self.C = C
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        bounds = [min(v, C) for v in [0, Z, 2 * Z, 3 * Z, 4 * Z, 4 * Z + 1, 4 * Z + 2, C]]
        self.groups = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        self.frames = frame_list(self.Y, self.X)
        self.gconv = nn.ModuleList([ortho(nn.Conv2d(b - a, GC, 1), R2) for a, b in self.groups])
        nraw = 2 * C + 4 * Z
        nin = len(self.groups) * len(self.frames) * GC + nraw
        self.s1 = ortho(nn.Conv2d(nin, COL, 1), R2)
        self.n0 = nn.GroupNorm(8, COL)
        self.s2 = ortho(nn.Conv2d(COL, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = ortho(nn.Conv2d(nraw, SKIP, 1), R2)
        self.nk = nn.GroupNorm(4, SKIP)

    def views(self, obs):
        B = obs.shape[0]
        Z, C, Y, X = self.Z, self.C, self.Y, self.X
        vb = obs.reshape(B, Y, X, C).permute(0, 3, 1, 2)
        vd = obs[:, :4 * Z].reshape(B, 4, Y, X, Z).permute(0, 1, 4, 2, 3).reshape(B, 4 * Z, Y, X)
        return torch.cat([obs, vb, vd], 1)

    def forward(self, obs):
        raw = self.views(obs)
        parts = []
        for (a, b), conv in zip(self.groups, self.gconv):
            h = F.relu(conv(obs[:, a:b]))
            for fr in self.frames:
                parts.append(to_frame(h, fr))
        parts.append(raw)
        s = F.relu(self.n0(self.s1(torch.cat(parts, 1))))
        s = F.relu(self.s2(s))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.nk(self.skip(raw)))
        return torch.cat([x, k], 1), x


class WindowHead(nn.Module):
    def __init__(self, cin, spec, mid, scale):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        K = max(max(a, b) for a, b in fps)
        self.frames = frame_list(spec.Y, spec.X)
        self.G = len(self.frames)
        self.T = len(fps)
        self.scale = scale
        self.pre = ortho(nn.Conv2d(cin, mid, 1), R2)
        self.win = ortho(nn.Conv2d(mid, 2 * self.G * self.T, 2 * K - 1, padding=K - 1), 0.01)
        self.mix = nn.Parameter(torch.zeros(2 * self.G))

    def forward(self, f):
        m = self.win(F.relu(self.pre(f)))
        B, _, H, W = m.shape
        m = m.reshape(B, 2, self.G, self.T, H, W)
        comps = []
        for gi, fr in enumerate(self.frames):
            a = to_frame(m[:, 0, gi], fr)
            comps.append(a.reshape(B, -1))
            b = to_frame(m[:, 1, gi], fr)
            comps.append(b.permute(0, 2, 3, 1).reshape(B, -1))
        L = torch.stack(comps, 1)
        w = F.softmax(self.scale * self.mix, dim=0)
        return (L * w.view(1, -1, 1)).sum(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = RemapTrunk(spec)
        self.head = WindowHead(WIDTH + SKIP, spec, MID, SCALE)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * WIDTH, GLOB), R2), nn.ReLU())
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.dense = ortho(nn.Linear(GLOB, spec.n_actions), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        f, x = self.trunk(obs)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        place = self.head(f)
        logits = torch.cat([place, self.done_head(g)], 1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
