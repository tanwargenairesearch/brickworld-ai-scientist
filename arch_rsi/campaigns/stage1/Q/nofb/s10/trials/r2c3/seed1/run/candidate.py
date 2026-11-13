import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HEADS = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.c1 = nn.Conv2d(spec.in_channels + 2, WIDTH, 3, padding=1)
        self.ln = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = F.relu(self.c1(torch.cat([obs, yy, xx], dim=1)))
        N = Y * X
        d = WIDTH // HEADS
        t = h.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.ln(t)).chunk(3, dim=-1)
        q = q.reshape(B, N, HEADS, d).transpose(1, 2)
        k = k.reshape(B, N, HEADS, d).transpose(1, 2)
        v = v.reshape(B, N, HEADS, d).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, WIDTH)
        t = t + self.proj(o)
        h = t.transpose(1, 2).reshape(B, WIDTH, Y, X)
        f = F.relu(self.c2(h))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
