import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
HEADS = 4

RECIPE = {"ppo_epochs": 2, "n_minibatches": 16, "ent_coef": 0.005}


class Attn(nn.Module):
    def __init__(self, c, heads, n):
        super().__init__()
        self.h = heads
        self.d = c // heads
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * self.h * self.d)
        self.out = nn.Linear(self.h * self.d, c)
        self.pos = nn.Parameter(torch.zeros(1, n, c))

    def forward(self, f):
        B, C, Y, X = f.shape
        t = f.flatten(2).transpose(1, 2)
        u = self.norm(t + self.pos)
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        q = q.reshape(B, -1, self.h, self.d).transpose(1, 2)
        k = k.reshape(B, -1, self.h, self.d).transpose(1, 2)
        v = v.reshape(B, -1, self.h, self.d).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, -1, self.h * self.d)
        t = t + self.out(o)
        return t.transpose(1, 2).reshape(B, C, Y, X)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
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
        n = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.attn = Attn(WIDTH, HEADS, n)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        f2 = F.relu(self.attn(f))
        place = self.head(f2)
        g = self.glob(torch.cat([f2.mean((2, 3)), f2.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
