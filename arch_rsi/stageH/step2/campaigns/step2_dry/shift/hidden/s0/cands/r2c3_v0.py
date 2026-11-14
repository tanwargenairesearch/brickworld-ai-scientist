import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 48
W = 40
HEADS = 4
FF = 80
PROJ = 32
HID = 32


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


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, t):
        B, N, d = t.shape
        h = self.n1(t)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, d)
        t = t + self.o(y)
        t = t + self.ff2(F.relu(self.ff1(self.n2(t))))
        return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.block = Block(W, HEADS, FF)
        self.head = PoolHead(W, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype)
        cy = ys.view(1, 1, Y, 1).expand(B, 1, Y, X)
        cx = xs.view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, cy, cx], dim=1))
        f = F.relu(s + self.c1(s))
        t = f.flatten(2).transpose(1, 2)
        t = self.block(t)
        f = F.relu(t.transpose(1, 2).reshape(B, -1, Y, X))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
