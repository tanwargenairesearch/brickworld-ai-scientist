import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "ent_coef": 0.02, "lr": 0.0008}

STEM = 64
COL = 48
D = 64
HEADS = 4
FF = 128
PROJ = 32
HID = 32
DONE_SCALE = 0.25


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
        Y = int(spec.Y)
        X = int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, D, 3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, D, Y, X) * 0.02)
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)
        self.head = PoolHead(D, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s)) + self.pos
        B, Dc, Y, X = x.shape
        N = Y * X
        dh = Dc // HEADS
        t = x.flatten(2).transpose(1, 2)
        h = self.ln1(t)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, N, HEADS, dh).transpose(1, 2)
        k = k.reshape(B, N, HEADS, dh).transpose(1, 2)
        v = v.reshape(B, N, HEADS, dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh), dim=-1)
        a = torch.matmul(att, v).transpose(1, 2).reshape(B, N, Dc)
        t = t + self.o(a)
        t = t + self.ff2(F.relu(self.ff1(self.ln2(t))))
        f = t.transpose(1, 2).reshape(B, Dc, Y, X)
        place = self.head(f)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        done = DONE_SCALE * self.done_head(g)
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
