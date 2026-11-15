import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1}

D = 64
HEADS = 4
LAYERS = 2


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 2 * d)
        self.ff2 = nn.Linear(2 * d, d)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        hd = d // self.h
        q = q.reshape(b, n, self.h, hd).transpose(1, 2)
        k = k.reshape(b, n, self.h, hd).transpose(1, 2)
        v = v.reshape(b, n, self.h, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.o(y)
        x = x + self.ff2(F.relu(self.ff1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec.Y * spec.X
        self.n_types = spec.n_types
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, D, 1), nn.ReLU(),
            nn.Conv2d(D, D, 1))
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS) for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.place = nn.Linear(D, spec.n_types)
        with torch.no_grad():
            self.place.weight.mul_(0.1)
            self.place.bias.zero_()
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        b = s.shape[0]
        t = s.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.lnf(t)
        place = self.place(t).transpose(1, 2).reshape(b, -1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
