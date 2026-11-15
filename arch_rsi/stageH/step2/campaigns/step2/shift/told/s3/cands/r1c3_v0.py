import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.02}

D = 64
HEADS = 4
LAYERS = 2
FF = 128
WIDTH = 48
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
    def __init__(self, Y, X):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)
        self.rel = nn.Parameter(torch.zeros(HEADS, (2 * Y - 1) * (2 * X - 1)))
        ys = torch.arange(Y).repeat_interleave(X)
        xs = torch.arange(X).repeat(Y)
        idx = (ys[:, None] - ys[None, :] + Y - 1) * (2 * X - 1) + (xs[:, None] - xs[None, :] + X - 1)
        self.register_buffer("idx", idx)

    def forward(self, h):
        B, N, _ = h.shape
        qkv = self.qkv(self.ln1(h)).view(B, N, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D // HEADS)
        att = att + self.rel[:, self.idx].unsqueeze(0)
        att = att.softmax(-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, N, D)
        h = h + self.o(y)
        h = h + self.ff2(F.relu(self.ff1(self.ln2(h))))
        return h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = spec.X
        self.Y = spec.Y
        n = spec.X * spec.Y
        self.embed = nn.Linear(spec.in_channels, D)
        self.pos = nn.Parameter(torch.randn(n, D) * 0.02)
        self.blocks = nn.ModuleList([Block(spec.Y, spec.X) for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.conv = nn.Conv2d(D, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        tok = obs.permute(0, 2, 3, 1).reshape(B, Y * X, C)
        h = self.embed(tok) + self.pos.unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        h = self.lnf(h)
        m = h.reshape(B, Y, X, D).permute(0, 3, 1, 2)
        f = F.relu(self.conv(m))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
