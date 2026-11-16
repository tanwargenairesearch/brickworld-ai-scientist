import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
HID = 32
GLOB = 64
SCALE = 5.0


def frame_tf(x, g):
    if g >= 4:
        x = x.transpose(2, 3)
    m = g % 4
    if m == 1:
        x = x.flip(3)
    elif m == 2:
        x = x.flip(2)
    elif m == 3:
        x = x.flip(2).flip(3)
    return x


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        for m in [self.stem1, self.stem2, self.c1, self.c2, self.skip]:
            ortho(m, math.sqrt(2))

    def forward(self, obs):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1), x


class WindowHead(nn.Module):
    def __init__(self, cin, fps, n_frames, scale):
        super().__init__()
        fl = [(int(a), int(b)) for a, b in fps]
        K = max(max(a, b) for a, b in fl)
        self.T = len(fl)
        self.G = n_frames
        self.scale = scale
        self.conv = nn.Conv2d(cin, HID, 2 * K - 1, padding=K - 1)
        ortho(self.conv, math.sqrt(2))
        self.out = nn.Conv2d(HID, self.G * self.T, 1)
        ortho(self.out, 0.01)
        self.alpha = nn.Parameter(torch.zeros(2 * self.G))

    def forward(self, f):
        B = f.shape[0]
        T = self.T
        m = self.out(F.relu(self.conv(f)))
        cands = []
        for g in range(self.G):
            mg = frame_tf(m[:, g * T:(g + 1) * T], g).flatten(2)
            cands.append(mg.reshape(B, -1))
            cands.append(mg.transpose(1, 2).reshape(B, -1))
        c = torch.stack(cands, 1)
        w = torch.softmax(self.scale * self.alpha, dim=0)
        return torch.einsum('bkp,k->bp', c, w)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        G = 8 if Y == X else 4
        self.trunk = Trunk(spec.in_channels)
        self.head = WindowHead(WIDTH + SKIP, spec.footprints, G, SCALE)
        self.glob = nn.Linear(2 * WIDTH, GLOB)
        ortho(self.glob, math.sqrt(2))
        self.done_head = nn.Linear(GLOB, 1)
        ortho(self.done_head, 0.01)
        self.value_head = nn.Linear(GLOB, 1)
        ortho(self.value_head, 1.0)
        self.dense = nn.Linear(GLOB, int(spec.n_place))
        ortho(self.dense, 0.01)

    def forward(self, obs, mask):
        f, x = self.trunk(obs)
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)))
        place = self.head(f) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
