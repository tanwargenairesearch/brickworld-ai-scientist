import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
MID = 32
GLOB = 64
FSCALE = 3.0
OSCALE = 5.0
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


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = ortho(nn.Conv2d(cin, STEM, 1), R2)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), R2)

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1), x


class PerTypeWindowHead(nn.Module):
    def __init__(self, cin, spec, mid):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        K = max(max(a, b) for a, b in fps)
        self.frames = frame_list(spec.Y, spec.X)
        self.G = len(self.frames)
        self.T = len(fps)
        self.pre = ortho(nn.Conv2d(cin, mid, 1), R2)
        self.win = ortho(nn.Conv2d(mid, 2 * self.G * self.T, 2 * K - 1, padding=K - 1), 0.01)
        self.fmix = nn.Parameter(torch.zeros(2, self.T, self.G))
        self.omix = nn.Parameter(torch.zeros(2))

    def forward(self, f):
        m = self.win(F.relu(self.pre(f)))
        B, _, H, W = m.shape
        m = m.reshape(B, 2, self.G, self.T, H, W)
        wf = F.softmax(FSCALE * self.fmix, dim=2)
        flat = []
        for o in range(2):
            acc = None
            for gi, fr in enumerate(self.frames):
                a = to_frame(m[:, o, gi], fr) * wf[o, :, gi].view(1, -1, 1, 1)
                acc = a if acc is None else acc + a
            if o == 0:
                flat.append(acc.reshape(B, -1))
            else:
                flat.append(acc.permute(0, 2, 3, 1).reshape(B, -1))
        wo = F.softmax(OSCALE * self.omix, dim=0)
        return wo[0] * flat[0] + wo[1] * flat[1]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.head = PerTypeWindowHead(WIDTH + SKIP, spec, MID)
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
