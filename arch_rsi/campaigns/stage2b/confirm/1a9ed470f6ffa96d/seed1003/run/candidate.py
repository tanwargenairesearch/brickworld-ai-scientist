import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}

WIDTH = 64
HID = 32
AD = 32
HEADS = 4


class FootHead(nn.Module):
    def __init__(self, c, hidden, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hidden, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hidden, 1, 1) for _ in self.fps])

    def forward(self, f):
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            g = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(conv(g))))
        return torch.cat(res, dim=1)


class CellAttn(nn.Module):
    def __init__(self, c, d, heads):
        super().__init__()
        self.h = heads
        self.d = d
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * d)
        self.out = nn.Linear(d, c)

    def forward(self, f):
        B, C, Y, X = f.shape
        t = f.flatten(2).transpose(1, 2)
        n = t.shape[1]
        dh = self.d // self.h
        q, k, v = self.qkv(self.norm(t)).chunk(3, dim=-1)
        q = q.reshape(B, n, self.h, dh).transpose(1, 2)
        k = k.reshape(B, n, self.h, dh).transpose(1, 2)
        v = v.reshape(B, n, self.h, dh).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, n, self.d)
        t = t + self.out(o)
        return t.transpose(1, 2).reshape(B, C, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.attn = CellAttn(WIDTH, AD, HEADS)
        self.place_head = FootHead(WIDTH, HID, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        f = self.body(torch.cat([obs, ys, xs], dim=1))
        f = self.attn(f)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
