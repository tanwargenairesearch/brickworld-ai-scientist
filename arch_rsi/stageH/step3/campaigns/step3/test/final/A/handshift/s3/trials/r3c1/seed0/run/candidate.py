import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
GLOB = 64
HID = 32
SCALE = 5.0
REINTERP = True
DIHEDRAL_IN = False
MASK_IN = False
R2 = 1.4142135623730951
PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def apply_frame(t, g):
    if g >= 4:
        t = t.transpose(-1, -2)
    k = g % 4
    if k == 1:
        t = t.flip(-1)
    elif k == 2:
        t = t.flip(-2)
    elif k == 3:
        t = t.flip(-2).flip(-1)
    return t


class InputBuilder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.C = int(spec.in_channels)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.Z = int(spec.Z)
        self.T = len(spec.footprints)
        self.n_place = int(spec.n_place)
        self.nf = 8 if self.Y == self.X else 4
        n = self.C * (len(PERMS) if REINTERP else 1)
        if DIHEDRAL_IN:
            n += (3 * self.Z + 1) * (self.nf - 1)
        if MASK_IN:
            n += self.T
        self.out_ch = n

    def forward(self, obs, mask):
        B = obs.shape[0]
        C, Y, X, Z = self.C, self.Y, self.X, self.Z
        parts = []
        if REINTERP:
            flat = obs.reshape(B, -1)
            s = (C, Y, X)
            for p in PERMS:
                t = flat.reshape(B, s[p[0]], s[p[1]], s[p[2]])
                parts.append(t.permute(0, p.index(0) + 1, p.index(1) + 1, p.index(2) + 1))
        else:
            parts.append(obs)
        if DIHEDRAL_IN:
            tg = torch.cat([obs[:, Z:4 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], 1)
            for g in range(1, self.nf):
                parts.append(apply_frame(tg, g))
        if MASK_IN:
            m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
            parts.append(m)
        return torch.cat(parts, 1)


class Trunk(nn.Module):
    def __init__(self, cin, width, skip):
        super().__init__()
        self.stem1 = ortho(nn.Conv2d(cin, STEM, 1), R2)
        self.stem2 = ortho(nn.Conv2d(STEM, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, width, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(4, width)
        self.c2 = ortho(nn.Conv2d(width, width, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(4, width)
        self.skip = ortho(nn.Conv2d(cin, skip, 1), 1.0)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * width, GLOB), R2), nn.ReLU())
        self.out_ch = width + skip

    def forward(self, xin):
        s = F.relu(self.stem1(xin))
        s = F.relu(self.stem2(s))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, self.skip(xin)], dim=1)
        return h, g


class WindowHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(fps)
        K = max(max(a, b) for a, b in fps)
        self.nf = 8 if self.Y == self.X else 4
        self.hid = ortho(nn.Conv2d(cin, HID, 1), R2)
        self.win = ortho(nn.Conv2d(HID, self.nf * self.T, 2 * K - 1, padding=K - 1), 0.01)
        self.alpha = nn.Parameter(torch.zeros(2 * self.nf))
        self.dense = ortho(nn.Linear(GLOB, self.T * self.Y * self.X), 0.01)

    def forward(self, h, g):
        B = h.shape[0]
        m = self.win(F.relu(self.hid(h)))
        m = m.reshape(B, self.nf, self.T, self.Y, self.X)
        outs = []
        for gi in range(self.nf):
            mg = apply_frame(m[:, gi], gi)
            outs.append(mg.reshape(B, -1))
            outs.append(mg.permute(0, 2, 3, 1).reshape(B, -1))
        L = torch.stack(outs, 1)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        place = (L * w.reshape(1, -1, 1)).sum(1)
        return place + self.dense(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.inp = InputBuilder(spec)
        self.trunk = Trunk(self.inp.out_ch, WIDTH, SKIP)
        self.head = WindowHead(self.trunk.out_ch, spec)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        xin = self.inp(obs, mask)
        h, g = self.trunk(xin)
        place = self.head(h, g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
