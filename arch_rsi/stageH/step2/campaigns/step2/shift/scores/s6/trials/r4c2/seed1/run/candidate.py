import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.0015, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

D = 48
HEADS = 4
FF = 96
LAYERS = 2


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.heads
        qkv = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        Y, X = int(spec.Y), int(spec.X)
        cin = int(spec.in_channels) + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, 64, 1), nn.ReLU(),
            nn.Conv2d(64, D, 1), nn.ReLU())
        self.local = nn.Conv2d(D, D, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, Y * X, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D)
        self.place = nn.Linear(D, self.nt)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nt * Y * X].float().reshape(B, self.nt, Y, X)
        s = self.stem(torch.cat([obs, m], dim=1))
        s = F.relu(s + self.local(s))
        t = s.flatten(2).transpose(1, 2) + self.pos
        for b in self.blocks:
            t = b(t)
        t = self.norm(t)
        place = self.place(t).transpose(1, 2).reshape(B, -1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
