import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "lr": 0.001, "gamma": 0.95, "gae_lambda": 0.95}

STEM = 64
D = 48
HEADS = 4
FF = 96
NL = 2
PROJ = 32
HID = 32
DK = 8


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.h
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.reshape(B, N, self.h, hd).transpose(1, 2)
        k = k.reshape(B, N, self.h, hd).transpose(1, 2)
        v = v.reshape(B, N, self.h, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


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
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        cells = self.Y * self.X
        na = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1))
        self.pos = nn.Parameter(torch.zeros(1, cells, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.head = PoolHead(D, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        k = max(1, min(DK, 200000 // (cells * na)))
        self.dproj = nn.Conv2d(D, k, 1)
        self.dense = nn.Linear(k * cells, na)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, yy, xx], dim=1))
        t = s.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.nf(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        d = self.dense(F.relu(self.dproj(f)).flatten(1))
        return logits + d, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
