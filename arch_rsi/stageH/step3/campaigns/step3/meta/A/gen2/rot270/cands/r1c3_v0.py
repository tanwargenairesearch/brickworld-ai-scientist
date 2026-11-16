import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 32
HID = 16
GLOB = 64
MIX_SCALE = 5.0
HYPS = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)]


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def flat(z, order):
    if order == 0:
        return z.flatten(1)
    if order == 1:
        return z.permute(0, 2, 3, 1).flatten(1)
    if order == 2:
        return z.permute(0, 1, 3, 2).flatten(1)
    return z.permute(0, 3, 2, 1).flatten(1)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = init_layer(nn.Conv2d(cin, STEM, 1))
        self.s2 = init_layer(nn.Conv2d(STEM, COL, 1))
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = init_layer(nn.Conv2d(cin, SKIP, 1))
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return f, torch.cat([f, k], 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.H = len(HYPS)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1))
        self.conv = init_layer(nn.Conv2d(self.H * PROJ, self.H * HID, self.K, groups=self.H))
        self.out = init_layer(nn.Conv2d(self.H * HID, self.H * self.T, 1, groups=self.H), 0.01)
        self.alpha = nn.Parameter(torch.zeros(self.H))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.dense = init_layer(nn.Linear(GLOB, spec.n_actions), 0.01)
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        f, h = self.trunk(obs)
        B = obs.shape[0]
        p = F.relu(self.proj(h))
        u = F.pad(p.repeat(1, self.H, 1, 1), (0, self.K - 1, 0, self.K - 1))
        z = self.out(F.relu(self.conv(u))).reshape(B, self.H, self.T, self.Y, self.X)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        fulls = []
        for i, (order, first) in enumerate(HYPS):
            pl = flat(z[:, i], order)
            if first == 1:
                fulls.append(torch.cat([d, pl], 1))
            else:
                fulls.append(torch.cat([pl, d], 1))
        full = torch.stack(fulls, 1)
        w = torch.softmax(MIX_SCALE * self.alpha, 0)
        logits = (full * w.view(1, -1, 1)).sum(1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
