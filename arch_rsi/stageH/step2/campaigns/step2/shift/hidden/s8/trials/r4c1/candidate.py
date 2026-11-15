import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 512, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "lr": 0.0015, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
HEADS = 4


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


class AttnBlock(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.h = heads
        self.d = w // heads
        self.pos = nn.Linear(2, w)
        self.ln1 = nn.LayerNorm(w)
        self.qkv = nn.Linear(w, 3 * w)
        self.o = nn.Linear(w, w)
        self.ln2 = nn.LayerNorm(w)
        self.ff1 = nn.Linear(w, 2 * w)
        self.ff2 = nn.Linear(2 * w, w)

    def forward(self, f):
        B, W, Y, X = f.shape
        N = Y * X
        ys = torch.linspace(-1.0, 1.0, Y, device=f.device, dtype=f.dtype)
        xs = torch.linspace(-1.0, 1.0, X, device=f.device, dtype=f.dtype)
        gy = ys.reshape(Y, 1).expand(Y, X)
        gx = xs.reshape(1, X).expand(Y, X)
        coords = torch.stack([gy, gx], dim=-1).reshape(1, N, 2)
        t = f.flatten(2).transpose(1, 2) + self.pos(coords)
        z = self.ln1(t)
        qkv = self.qkv(z).reshape(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d), dim=-1)
        a = torch.matmul(att, v).transpose(1, 2).reshape(B, N, W)
        t = t + self.o(a)
        t = t + self.ff2(F.relu(self.ff1(self.ln2(t))))
        return t.transpose(1, 2).reshape(B, W, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.attn = AttnBlock(WIDTH, HEADS)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        x = F.relu(x + self.c2(x))
        f = self.attn(x)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
