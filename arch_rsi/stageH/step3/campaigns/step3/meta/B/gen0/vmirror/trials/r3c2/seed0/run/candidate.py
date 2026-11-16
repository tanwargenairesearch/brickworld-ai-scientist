import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "gae_lambda": 0.95}

D = 48
HEADS = 4
FF = 96
LAYERS = 2
TOK = 4


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, d = x.shape
        hd = d // self.h
        qkv = self.qkv(self.n1(x)).view(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, d)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.T = len(spec.footprints)
        self.inp = nn.Linear(int(spec.in_channels), D)
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(LAYERS)])
        self.nf = nn.LayerNorm(D)
        self.loc = nn.Linear(D, self.T)
        self.tok = nn.Linear(D, TOK)
        self.flat = nn.Linear(TOK * n, int(spec.n_place))
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        t = obs.flatten(2).transpose(1, 2)
        h = self.inp(t) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.nf(h)
        loc = self.loc(h).transpose(1, 2).reshape(B, -1)
        dense = self.flat(F.relu(self.tok(h)).flatten(1))
        place = loc + dense
        g = self.glob(torch.cat([h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
