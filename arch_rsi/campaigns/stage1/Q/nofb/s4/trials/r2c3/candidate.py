import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HEADS = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.conv = nn.Conv2d(W, W, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, int(spec.Y) * int(spec.X), W))
        nn.init.normal_(self.pos, std=0.02)
        self.ln1 = nn.LayerNorm(W)
        self.qkv = nn.Linear(W, 3 * W)
        self.proj = nn.Linear(W, W)
        self.ln2 = nn.LayerNorm(W)
        self.mlp = nn.Sequential(nn.Linear(W, 2 * W), nn.ReLU(), nn.Linear(2 * W, W))
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        f = F.relu(f + self.conv(f))
        B, C, Y, X = f.shape
        N = Y * X
        D = C // HEADS
        t = f.flatten(2).transpose(1, 2) + self.pos
        h = self.ln1(t)
        qkv = self.qkv(h).reshape(B, N, 3, HEADS, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, C)
        t = t + self.proj(o)
        t = t + self.mlp(self.ln2(t))
        f = t.transpose(1, 2).reshape(B, C, Y, X)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
