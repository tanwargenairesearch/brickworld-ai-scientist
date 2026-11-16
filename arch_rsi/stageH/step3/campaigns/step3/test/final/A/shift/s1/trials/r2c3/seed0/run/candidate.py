import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

GC = 4
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


def flatten_order(a, o):
    B = a.shape[0]
    if o == 0:
        return a.reshape(B, -1)
    if o == 1:
        return a.permute(0, 2, 3, 1).reshape(B, -1)
    return a.permute(0, 2, 1, 3).reshape(B, -1)


class GroupFrameTrunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        C = int(spec.in_channels)
        bounds = [min(v, C) for v in [0, Z, 2 * Z, 3 * Z, 4 * Z, 4 * Z + 1, 4 * Z + 2, C]]
        self.groups = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        self.frames = frame_list(spec.Y, spec.X)
        self.gconv = nn.ModuleList([ortho(nn.Conv2d(b - a, GC, 1), R2) for a, b in self.groups])
        nin = len(self.groups) * len(self.frames) * GC
        self.s2 = ortho(nn.Conv2d(nin, COL, 1), R2)
        self.n0 = nn.GroupNorm(8, COL)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = ortho(nn.Conv2d(C, SKIP, 1), R2)
        self.nk = nn.GroupNorm(4, SKIP)

    def forward(self, obs):
        parts = []
        for (a, b), conv in zip(self.groups, self.gconv):
            h = F.relu(conv(obs[:, a:b]))
            for fr in self.frames:
                parts.append(to_frame(h, fr))
        s = F.relu(self.n0(self.s2(torch.cat(parts, 1))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.nk(self.skip(obs)))
        return torch.cat([x, k], 1), x


class LayoutHead(nn.Module):
    def __init__(self, cin, spec, mid, scale, n_done):
        super().__init__()
        self.frames = frame_list(spec.Y, spec.X)
        self.G = len(self.frames)
        self.T = len(spec.footprints)
        self.O = 3
        self.D = n_done
        self.scale = scale
        self.pre = ortho(nn.Conv2d(cin, mid, 3, padding=1), R2)
        self.out = ortho(nn.Conv2d(mid, self.D * self.O * self.G * self.T, 1), 0.01)
        self.mix = nn.Parameter(torch.zeros(self.D * self.O * self.G))

    def forward(self, f, done):
        h = F.relu(self.pre(f))
        m = self.out(h)
        B, _, H, W = m.shape
        m = m.reshape(B, self.D, self.O, self.G, self.T, H, W)
        comps = []
        for d in range(self.D):
            for o in range(self.O):
                for gi in range(self.G):
                    p = flatten_order(to_frame(m[:, d, o, gi], self.frames[gi]), o)
                    if d == 0:
                        comps.append(torch.cat([p, done], 1))
                    else:
                        comps.append(torch.cat([done, p], 1))
        L = torch.stack(comps, 1)
        w = F.softmax(self.scale * self.mix, dim=0)
        return (L * w.view(1, -1, 1)).sum(1), h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = GroupFrameTrunk(spec)
        self.head = LayoutHead(WIDTH + SKIP, spec, MID, SCALE, 1)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * WIDTH, GLOB), R2), nn.ReLU())
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.dense = ortho(nn.Linear(GLOB, spec.n_actions), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        f, x = self.trunk(obs)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        logits, _ = self.head(f, self.done_head(g))
        logits = logits + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
