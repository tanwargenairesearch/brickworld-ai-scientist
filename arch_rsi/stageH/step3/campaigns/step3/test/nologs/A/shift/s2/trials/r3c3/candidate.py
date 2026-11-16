import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16}

STEM = 48
D = 32
NH = 4
FF = 64
NL = 2
GLOB = 64


class Block(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.nh
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, N, self.nh, hd).transpose(1, 2)
        k = k.reshape(B, N, self.nh, hd).transpose(1, 2)
        v = v.reshape(B, N, self.nh, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.n_types = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 3, padding=1))
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, NH, FF) for _ in range(NL)])
        self.lnf = nn.LayerNorm(D)
        self.out = nn.Linear(D, self.n_types)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        t = s.flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            t = b(t)
        t = self.lnf(t)
        place = self.out(t).transpose(1, 2).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
