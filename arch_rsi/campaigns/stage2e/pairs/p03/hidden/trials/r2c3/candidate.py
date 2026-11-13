import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.98}

WIDTH = 64
HEADS = 4


class AttnBlock(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(w)
        self.qkv = nn.Linear(w, 3 * w)
        self.out = nn.Linear(w, w)
        self.ln2 = nn.LayerNorm(w)
        self.mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.ReLU(), nn.Linear(2 * w, w))

    def forward(self, x):
        b, n, w = x.shape
        d = w // self.h
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.reshape(b, n, self.h, d).transpose(1, 2)
        k = k.reshape(b, n, self.h, d).transpose(1, 2)
        v = v.reshape(b, n, self.h, d).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(b, n, w)
        x = x + self.out(o)
        x = x + self.mlp(self.ln2(x))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels + 2
        self.body = nn.Sequential(
            nn.Conv2d(c, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.attn = AttnBlock(WIDTH, HEADS)
        self.place_head = nn.Sequential(
            nn.Conv2d(2 * WIDTH, WIDTH, 1), nn.ReLU(), nn.Conv2d(WIDTH, spec.n_types, 1))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        b, _, yy, xx = obs.shape
        ys = torch.linspace(-1.0, 1.0, yy, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, xx, device=obs.device, dtype=obs.dtype)
        cy = ys.view(1, 1, yy, 1).expand(b, 1, yy, xx)
        cx = xs.view(1, 1, 1, xx).expand(b, 1, yy, xx)
        f = self.body(torch.cat([obs, cy, cx], dim=1))
        t = f.flatten(2).transpose(1, 2)
        t = self.attn(t)
        a = t.transpose(1, 2).reshape(b, WIDTH, yy, xx)
        place = self.place_head(torch.cat([f, a], dim=1)).flatten(1)
        g = self.glob(torch.cat([a.mean((2, 3)), a.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
