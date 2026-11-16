import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}

D = 48
HEADS = 4
FF = 96
LAYERS = 2
PH = 48


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, x):
        b, n, d = x.shape
        hd = d // self.h
        qkv = self.qkv(self.ln1(x)).reshape(b, n, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.o(y)
        x = x + self.ff(self.ln2(x))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_cells = int(spec.Y) * int(spec.X)
        self.n_types = int(spec.n_types)
        self.embed = nn.Conv2d(spec.in_channels, D, 3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, n_cells, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.ph = nn.Sequential(nn.Linear(D, PH), nn.ReLU(), nn.Linear(PH, self.n_types))
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        x = self.embed(obs).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        place = self.ph(x).transpose(1, 2).reshape(b, -1)
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
