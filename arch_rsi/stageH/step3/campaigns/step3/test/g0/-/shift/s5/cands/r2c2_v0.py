import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2}

D = 64
NH = 4
NL = 2
FF = 128


class Block(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.nh = nh
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, t):
        B, N, d = t.shape
        h = self.n1(t)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        hd = d // self.nh
        q = q.reshape(B, N, self.nh, hd).transpose(1, 2)
        k = k.reshape(B, N, self.nh, hd).transpose(1, 2)
        v = v.reshape(B, N, self.nh, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / math.sqrt(hd), dim=-1)
        u = torch.matmul(a, v).transpose(1, 2).reshape(B, N, d)
        t = t + self.o(u)
        t = t + self.f2(F.relu(self.f1(self.n2(t))))
        return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.yx = int(spec.Y) * int(spec.X)
        self.nt = int(spec.n_types)
        self.embed = nn.Linear(int(spec.in_channels), D)
        self.pos = nn.Parameter(torch.randn(1, self.yx, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, NH, FF) for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.place = nn.Linear(D, self.nt)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        tok = obs.flatten(2).transpose(1, 2)
        t = self.embed(tok) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.nf(t)
        pl = self.place(t).transpose(1, 2).reshape(B, self.nt * self.yx)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([pl, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
