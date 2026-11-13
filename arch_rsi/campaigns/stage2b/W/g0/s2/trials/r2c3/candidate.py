import math
import torch
import torch.nn as nn
import torch.nn.functional as F

COL = 64
WIDTH = 48
HEADS = 4
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3}


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, N, D = x.shape
        dh = D // self.h
        y = self.norm(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        q = q.reshape(B, N, self.h, dh).transpose(1, 2)
        k = k.reshape(B, N, self.h, dh).transpose(1, 2)
        v = v.reshape(B, N, self.h, dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, D)
        return x + self.out(o)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, COL, 1), nn.ReLU())
        self.body = nn.Sequential(
            nn.Conv2d(COL, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.attn = Attn(WIDTH, HEADS)
        self.place_head = nn.Conv2d(WIDTH + COL, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * (WIDTH + COL), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        c = self.col(obs)
        b = self.body(c)
        t = b.flatten(2).transpose(1, 2)
        t = self.attn(t)
        b = t.transpose(1, 2).reshape(B, WIDTH, Y, X)
        f = torch.cat([b, c], dim=1)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
