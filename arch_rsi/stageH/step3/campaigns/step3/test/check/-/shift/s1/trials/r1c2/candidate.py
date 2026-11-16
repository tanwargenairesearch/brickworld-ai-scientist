import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "ppo_epochs": 2}

D = 48
HEADS = 4
FF = 96
LAYERS = 2
PROJ = 32
HID = 32


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.nh = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, t):
        b, n, d = t.shape
        hd = d // self.nh
        qkv = self.qkv(self.ln1(t)).view(b, n, 3, self.nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(b, n, d)
        t = t + self.proj(o)
        t = t + self.ff2(F.gelu(self.ff1(self.ln2(t))))
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = spec.X
        self.Y = spec.Y
        self.embed = nn.Linear(spec.in_channels + spec.X + spec.Y, D)
        self.blocks = nn.ModuleList([Block(D, HEADS, FF) for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.head = PoolHead(D, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = self.Y
        X = self.X
        ex = torch.eye(X, device=obs.device, dtype=obs.dtype).view(1, X, 1, X).expand(B, X, Y, X)
        ey = torch.eye(Y, device=obs.device, dtype=obs.dtype).view(1, Y, Y, 1).expand(B, Y, Y, X)
        h = torch.cat([obs, ex, ey], dim=1)
        t = h.flatten(2).transpose(1, 2)
        t = self.embed(t)
        for blk in self.blocks:
            t = blk(t)
        t = self.lnf(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
