import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0005, "ent_coef": 0.005, "gamma": 0.97}

D = 48
NH = 4
FF = 96
NL = 2
STEM = 32


class Block(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.nh = nh
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, t):
        B, N, Dm = t.shape
        hd = Dm // self.nh
        qkv = self.qkv(self.n1(t)).reshape(B, N, 3, self.nh, hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        t = t + self.proj(o)
        return t + self.ff(self.n2(t))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 3, padding=1))
        self.pos = nn.Parameter(torch.randn(1, Y * X, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, NH, FF) for _ in range(NL)])
        self.norm = nn.LayerNorm(D)
        self.place = nn.Linear(D, spec.n_types)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        t = s.flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            t = b(t)
        t = self.norm(t)
        place = self.place(t).transpose(1, 2).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
