import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
GLOB = 64
HID = 32
KF = 8
SCALE = 5.0
R2 = 1.4142135623730951


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


class Trunk(nn.Module):
    def __init__(self, spec, width, skip):
        super().__init__()
        c = spec.in_channels
        self.stem1 = ortho(nn.Conv2d(c, STEM, 1), R2)
        self.stem2 = ortho(nn.Conv2d(STEM, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, width, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(4, width)
        self.c2 = ortho(nn.Conv2d(width, width, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(4, width)
        self.skip = ortho(nn.Conv2d(c, skip, 1), 1.0)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * width, GLOB), R2), nn.ReLU())
        self.out_ch = width + skip

    def forward(self, obs):
        s = F.relu(self.stem1(obs))
        s = F.relu(self.stem2(s))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, self.skip(obs)], dim=1)
        return h, g


class LayoutHead(nn.Module):
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
        self.alpha = nn.Parameter(torch.zeros(3 * self.nf))

    def forward(self, h):
        B = h.shape[0]
        m = self.win(F.relu(self.hid(h)))
        m = m.reshape(B, self.nf, self.T, self.Y, self.X)
        outs = []
        for gi in range(self.nf):
            mg = apply_frame(m[:, gi], gi)
            outs.append(mg.reshape(B, -1))
            outs.append(mg.permute(0, 2, 1, 3).reshape(B, -1))
            outs.append(mg.permute(0, 2, 3, 1).reshape(B, -1))
        L = torch.stack(outs, 1)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        return (L * w.reshape(1, -1, 1)).sum(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec, WIDTH, SKIP)
        YX = int(spec.Y) * int(spec.X)
        A = int(spec.n_actions)
        self.head = LayoutHead(self.trunk.out_ch, spec)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.beta = nn.Parameter(torch.zeros(2))
        self.fproj = ortho(nn.Conv2d(self.trunk.out_ch, KF, 1), R2)
        self.remap = ortho(nn.Linear(KF * YX + GLOB, A), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        place = self.head(h)
        done = self.done_head(g)
        wd = torch.softmax(SCALE * self.beta, dim=0)
        logits = wd[0] * torch.cat([place, done], dim=1) + wd[1] * torch.cat([done, place], dim=1)
        u = F.relu(self.fproj(h)).flatten(1)
        logits = logits + self.remap(torch.cat([u, g], dim=1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
