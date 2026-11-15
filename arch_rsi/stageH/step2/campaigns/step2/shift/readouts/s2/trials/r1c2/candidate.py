import torch
import torch.nn as nn
import torch.nn.functional as F
import math

D = 48
HEADS = 4
FF = 96
NLAYERS = 2
WIDTH = 40
PROJ = 32
HID = 32
NFREQ = 2


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, t):
        B, N, Dm = t.shape
        hd = Dm // self.h
        q, k, v = self.qkv(self.ln1(t)).chunk(3, dim=-1)
        q = q.reshape(B, N, self.h, hd).transpose(1, 2)
        k = k.reshape(B, N, self.h, hd).transpose(1, 2)
        v = v.reshape(B, N, self.h, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        t = t + self.o(y)
        t = t + self.ff(self.ln2(t))
        return t


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


def coord_channels(Y, X):
    ys = torch.arange(Y, dtype=torch.float32).view(Y, 1).expand(Y, X)
    xs = torch.arange(X, dtype=torch.float32).view(1, X).expand(Y, X)
    chans = []
    for k in range(1, NFREQ + 1):
        ax = 2.0 * math.pi * k * xs / X
        ay = 2.0 * math.pi * k * ys / Y
        chans += [torch.sin(ax), torch.cos(ax), torch.sin(ay), torch.cos(ay)]
    return torch.stack(chans, 0).unsqueeze(0)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = spec.Y
        self.X = spec.X
        self.register_buffer('coords', coord_channels(spec.Y, spec.X))
        ncoord = 4 * NFREQ
        self.stem = nn.Conv2d(spec.in_channels + ncoord, D, 1)
        self.pos = nn.Parameter(torch.randn(1, spec.Y * spec.X, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(NLAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.conv = nn.Conv2d(D, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        c = self.coords.expand(B, -1, -1, -1)
        s = self.stem(torch.cat([obs, c], dim=1))
        t = s.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.lnf(t)
        f = t.transpose(1, 2).reshape(B, D, self.Y, self.X)
        f = F.relu(self.conv(f))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
