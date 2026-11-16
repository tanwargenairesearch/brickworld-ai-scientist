import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.01}

D = 48
HEADS = 4
FF = 96
G = 64


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.h
        qkv = self.qkv(self.n1(x)).reshape(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        ncell = int(spec.Y) * int(spec.X)
        self.nt = int(spec.n_types)
        self.stem = nn.Conv2d(spec.in_channels, D, 3, padding=1)
        self.stem2 = nn.Conv2d(D, D, 1)
        self.pos = nn.Parameter(torch.randn(ncell, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(2)])
        self.nf = nn.LayerNorm(D)
        self.place = nn.Linear(D, self.nt)
        nn.init.normal_(self.place.weight, std=0.01)
        nn.init.zeros_(self.place.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = F.relu(self.stem(obs))
        s = self.stem2(s)
        t = s.flatten(2).transpose(1, 2) + self.pos.unsqueeze(0)
        for blk in self.blocks:
            t = blk(t)
        t = self.nf(t)
        place = self.place(t).transpose(1, 2).reshape(B, -1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
