import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "ppo_epochs": 1, "n_minibatches": 8, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

D = 64
HEADS = 4
NL = 2


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 2 * d)
        self.f2 = nn.Linear(2 * d, d)

    def forward(self, x):
        B = x.shape[0]
        N = x.shape[1]
        Dm = x.shape[2]
        hd = Dm // self.h
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.reshape(B, N, self.h, hd).transpose(1, 2)
        k = k.reshape(B, N, self.h, hd).transpose(1, 2)
        v = v.reshape(B, N, self.h, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.proj(o)
        x = x + self.f2(F.gelu(self.f1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.nt = int(spec.n_types)
        self.embed = nn.Linear(spec.in_channels, D)
        self.pos = nn.Parameter(torch.zeros(1, n, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS) for _ in range(NL)])
        self.ln = nn.LayerNorm(D)
        self.place = nn.Linear(D, self.nt)
        self.omix = nn.Linear(n, n)
        nn.init.normal_(self.omix.weight, std=0.01)
        nn.init.zeros_(self.omix.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        t = obs.flatten(2).transpose(1, 2)
        x = self.embed(t) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        p = self.place(x).transpose(1, 2)
        p = p + self.omix(p)
        place = p.flatten(1)
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
