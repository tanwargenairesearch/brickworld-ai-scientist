import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 16
HID = 16
GLOB = 64
MIX_SCALE = 5.0
NO = 4


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def frame(u, k):
    if k >= 4:
        u = u.transpose(2, 3)
    if k % 2 == 1:
        u = u.flip(3)
    if (k // 2) % 2 == 1:
        u = u.flip(2)
    return u


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
        self.W = 2 * self.K - 1
        self.nf = 8 if self.X == self.Y else 4
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1))
        self.conv = init_layer(nn.Conv2d(self.nf * PROJ, self.nf * HID, self.W, padding=self.K - 1, groups=self.nf))
        self.out = init_layer(nn.Conv2d(self.nf * HID, self.nf * NO * self.T, 1, groups=self.nf), 0.01)
        self.alpha = nn.Parameter(torch.zeros(self.nf * NO))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.dense = init_layer(nn.Linear(GLOB, spec.n_actions), 0.01)
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        f, h = self.trunk(obs)
        B = obs.shape[0]
        p = F.relu(self.proj(h))
        u = torch.cat([frame(p, k) for k in range(self.nf)], 1)
        z = self.out(F.relu(self.conv(u)))
        z = z.reshape(B, self.nf, NO, self.T, self.Y, self.X)
        tyx = z.flatten(3)
        yxt = z.permute(0, 1, 2, 4, 5, 3).flatten(3)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        dd = d.view(B, 1, 1).expand(B, self.nf, 1)
        h0 = torch.cat([tyx[:, :, 0], dd], 2)
        h1 = torch.cat([yxt[:, :, 1], dd], 2)
        h2 = torch.cat([dd, tyx[:, :, 2]], 2)
        h3 = torch.cat([dd, yxt[:, :, 3]], 2)
        full = torch.stack([h0, h1, h2, h3], 2).reshape(B, self.nf * NO, -1)
        w = torch.softmax(MIX_SCALE * self.alpha, 0)
        logits = (full * w.view(1, -1, 1)).sum(1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
