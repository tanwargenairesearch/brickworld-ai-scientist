import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
COL = 48
HEADS = 4
DH = 8
WIDTH = 40
PROJ = 32
HID = 32
BIAS_SCALE = 10.0


def small_init(layer):
    nn.init.orthogonal_(layer.weight, 0.01)
    nn.init.zeros_(layer.bias)


class AlignAttn(nn.Module):
    def __init__(self, dq, din, heads, dh, Y, X):
        super().__init__()
        self.h = heads
        self.dh = dh
        self.norm = nn.LayerNorm(dq)
        self.qk = nn.Linear(dq, 2 * heads * dh)
        self.v = nn.Linear(din, heads * dh)
        self.out = nn.Linear(heads * dh, heads * dh)
        self.bias = nn.Parameter(torch.zeros(heads, (2 * Y - 1) * X))
        ys = torch.arange(Y).repeat_interleave(X)
        xs = torch.arange(X).repeat(Y)
        dy = ys[None, :] - ys[:, None] + (Y - 1)
        dx = torch.remainder(xs[None, :] - xs[:, None], X)
        self.register_buffer('idx', dy * X + dx)

    def forward(self, t, o):
        B, N, d = t.shape
        qk = self.qk(self.norm(t)).reshape(B, N, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(o).reshape(B, N, self.h, self.dh).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att + BIAS_SCALE * self.bias[:, self.idx].unsqueeze(0)
        att = att.softmax(-1)
        r = torch.matmul(att, v).transpose(1, 2).reshape(B, N, self.h * self.dh)
        return self.out(r)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            small_init(o)

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
        cin = int(spec.in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.attn = AlignAttn(COL, cin, HEADS, DH, int(spec.Y), int(spec.X))
        self.c1 = nn.Conv2d(COL + HEADS * DH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        small_init(self.done_head)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        B, C, Y, X = s.shape
        t = s.flatten(2).transpose(1, 2)
        o = obs.flatten(2).transpose(1, 2)
        r = self.attn(t, o)
        a = r.transpose(1, 2).reshape(B, HEADS * DH, Y, X)
        x = F.relu(self.c1(torch.cat([s, a], dim=1)))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
