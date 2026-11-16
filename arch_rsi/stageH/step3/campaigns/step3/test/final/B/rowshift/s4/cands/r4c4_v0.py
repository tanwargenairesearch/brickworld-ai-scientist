import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

D = 40
HEADS = 4
FF = 80
LAYERS = 2
GLOB = 64


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, d = x.shape
        hd = d // self.h
        qkv = self.qkv(self.ln1(x)).reshape(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), -1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, d)
        x = x + self.proj(o)
        x = x + self.f2(F.relu(self.f1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_act = int(spec.n_actions)
        cells = self.Y * self.X
        self.cells = cells
        self.embed = nn.Linear(self.C, D)
        self.pos = nn.Parameter(torch.randn(cells, D) * 0.2)
        self.mask_tok = nn.Linear(self.n_act, D)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.wk = nn.Linear(D, D)
        self.wv = nn.Linear(D, D)
        self.aq = nn.Parameter(torch.randn(self.n_place, D) / math.sqrt(D))
        self.au = nn.Parameter(torch.randn(self.n_place, D) * 0.01)
        self.ab = nn.Parameter(torch.zeros(self.n_place))
        self.glob = nn.Linear(2 * D, GLOB)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        t = obs.reshape(B, self.C, self.cells).transpose(1, 2)
        h = self.embed(t) + self.pos
        mt = self.mask_tok(mask.float()).unsqueeze(1)
        h = torch.cat([h, mt], 1)
        for blk in self.blocks:
            h = blk(h)
        h = self.lnf(h)
        k = self.wk(h)
        v = self.wv(h)
        sc = torch.matmul(self.aq, k.transpose(1, 2)) / math.sqrt(D)
        a = torch.softmax(sc, -1)
        ctx = torch.matmul(a, v)
        place = (ctx * self.au).sum(-1) + self.ab
        g = F.relu(self.glob(torch.cat([h.mean(1), h.amax(1)], 1)))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
