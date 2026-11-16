import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.01, "gamma": 0.97, "lr": 0.0007}

STEM = 64
D = 48
HEADS = 4
FF = 96
LAYERS = 2


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
        q, k, v = qkv.unbind(0)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        cin = int(spec.in_channels) + 2 * self.T
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, D, 3, padding=1), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, self.Y * self.X, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(LAYERS)])
        self.nf = nn.LayerNorm(D)
        self.out = nn.Linear(D, self.T)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = self.Y
        X = self.X
        T = self.T
        m = mask[:, :T * Y * X].float()
        m1 = m.reshape(B, T, Y, X)
        m2 = m.reshape(B, T, X, Y).transpose(2, 3)
        f = self.stem(torch.cat([obs, m1, m2], dim=1))
        tok = f.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.nf(tok)
        place = self.out(tok).transpose(1, 2).reshape(B, T * Y * X)
        g = self.glob(torch.cat([tok.mean(1), tok.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
