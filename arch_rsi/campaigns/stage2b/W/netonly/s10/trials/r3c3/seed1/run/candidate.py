import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
BLOCKS = 2
HEADS = 4


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class CellAttn(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.h = heads
        self.d = w // heads
        self.norm1 = nn.LayerNorm(w)
        self.qkv = nn.Linear(w, 3 * w)
        self.out = nn.Linear(w, w)
        self.norm2 = nn.LayerNorm(w)
        self.ff1 = nn.Linear(w, 2 * w)
        self.ff2 = nn.Linear(2 * w, w)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)

    def forward(self, f):
        B, C, Y, X = f.shape
        N = Y * X
        x = f.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.reshape(B, N, self.h, self.d).transpose(1, 2)
        k = k.reshape(B, N, self.h, self.d).transpose(1, 2)
        v = v.reshape(B, N, self.h, self.d).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, C)
        x = x + self.out(o)
        x = x + self.ff2(F.relu(self.ff1(self.norm2(x))))
        return x.transpose(1, 2).reshape(B, C, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.attn = CellAttn(WIDTH, HEADS)
        self.type_heads = nn.ModuleList([nn.Conv2d(4 * WIDTH, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.attn(self.blocks(self.stem(obs)))
        outs = []
        for t, (dy, dx) in enumerate(self.footprints):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            h = torch.cat([f, mx, mn, av], dim=1)
            outs.append(self.type_heads[t](h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
