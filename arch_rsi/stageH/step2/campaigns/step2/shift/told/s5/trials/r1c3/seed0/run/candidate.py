import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
D = 48
HEADS = 4
WIDTH = 40
PROJ = 32
HID = 32


class RelAttn(nn.Module):
    def __init__(self, d, heads, Y, X):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(heads, Y * X))
        ys = torch.arange(Y).repeat_interleave(X)
        xs = torch.arange(X).repeat(Y)
        dy = (ys[None, :] - ys[:, None]) % Y
        dx = (xs[None, :] - xs[:, None]) % X
        self.register_buffer("idx", dy * X + dx)

    def forward(self, t):
        B, N, d = t.shape
        qkv = self.qkv(t).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        att = att + self.bias[:, self.idx].unsqueeze(0)
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, d)
        return self.out(o)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1), nn.ReLU())
        self.norm = nn.LayerNorm(D)
        self.attn = RelAttn(D, HEADS, spec.Y, spec.X)
        self.c1 = nn.Conv2d(D, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        B, C, Y, X = s.shape
        t = s.flatten(2).transpose(1, 2)
        t = t + self.attn(self.norm(t))
        s = t.transpose(1, 2).reshape(B, C, Y, X)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
