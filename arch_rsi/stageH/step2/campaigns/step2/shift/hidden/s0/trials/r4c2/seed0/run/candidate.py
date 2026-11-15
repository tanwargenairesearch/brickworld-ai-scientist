import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
HEADS = 4
GD = 64


class CellAttn(nn.Module):
    def __init__(self, c, ny, nx, heads):
        super().__init__()
        self.h = int(heads)
        self.d = c // self.h
        self.pos = nn.Parameter(torch.randn(1, ny * nx, c) * 0.02)
        self.ln = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * self.h * self.d)
        self.out = nn.Linear(self.h * self.d, c)

    def forward(self, f):
        b, c, yy, xx = f.shape
        n = yy * xx
        t = f.flatten(2).transpose(1, 2)
        z = self.ln(t + self.pos)
        qkv = self.qkv(z).reshape(b, n, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(b, n, self.h * self.d)
        t = t + self.out(o)
        return t.transpose(1, 2).reshape(b, c, yy, xx)


class PoolHead(nn.Module):
    def __init__(self, c, gd, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.gproj = nn.Linear(gd, proj)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f, g):
        p = self.proj(f) + self.gproj(g).unsqueeze(2).unsqueeze(3)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.attn = CellAttn(WIDTH, spec.Y, spec.X, HEADS)
        self.head = PoolHead(WIDTH, GD, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GD), nn.ReLU())
        self.done_head = nn.Linear(GD, 1)
        self.value_head = nn.Linear(GD, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        f = self.attn(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(f, g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
