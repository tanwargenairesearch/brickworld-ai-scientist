import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 4, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}

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


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, t):
        B, N, D = t.shape
        hd = D // self.h
        qkv = self.qkv(self.ln1(t)).reshape(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        t = t + self.o(y)
        return t + self.mlp(self.ln2(t))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        A = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, WIDTH, Y, X) * 0.02)
        self.block = Block(WIDTH, HEADS)
        self.head = PoolHead(2 * WIDTH, spec.footprints, PROJ, HID)
        self.aq = nn.Parameter(torch.randn(A, WIDTH) * 0.1)
        self.ak = nn.Linear(WIDTH, WIDTH)
        self.av = nn.Linear(WIDTH, WIDTH)
        self.aout = nn.Linear(WIDTH, 1)
        self.glob = nn.Sequential(nn.Linear(3 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        B, C, Y, X = f.shape
        t = (f + self.pos).flatten(2).transpose(1, 2)
        t = self.block(t)
        g2 = t.transpose(1, 2).reshape(B, C, Y, X)
        place = self.head(torch.cat([f, g2], dim=1))
        k = self.ak(t)
        v = self.av(t)
        sc = torch.einsum('ad,bnd->ban', self.aq, k) / math.sqrt(C)
        att = torch.softmax(sc, dim=-1)
        o = torch.einsum('ban,bnd->bad', att, v)
        place = place + self.aout(F.relu(o)).squeeze(-1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), t.mean(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
