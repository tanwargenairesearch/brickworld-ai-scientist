import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.0015, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 48
WIDTH = 32
PROJ = 4
HID = 24
HEADS = 4
FF = 64


def views(p, square):
    vs = [p, p.flip(2), p.flip(3), p.flip(2).flip(3)]
    if square:
        vs = vs + [v.transpose(2, 3) for v in vs]
    return torch.cat(vs, dim=1)


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
        self.square = X == Y
        nv = 8 if self.square else 4
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, Y * X, WIDTH))
        nn.init.normal_(self.pos, std=0.02)
        self.block = Block(WIDTH, HEADS, FF)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.h1 = nn.Conv2d(nv * PROJ + self.nt, HID, 3, padding=1)
        self.out = nn.Conv2d(HID, self.nt, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nt * Y * X].float().reshape(B, self.nt, Y, X)
        s = self.stem(obs)
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        t = f.flatten(2).transpose(1, 2) + self.pos
        t = self.block(t)
        a = t.transpose(1, 2).reshape(B, -1, Y, X)
        p = F.relu(self.proj(a))
        h = torch.cat([views(p, self.square), m], dim=1)
        h = F.relu(self.h1(h))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([a.mean((2, 3)), a.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
