import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "ppo_epochs": 2, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.01}

W = 32
SK = 24
HID = 24
NBLK = 1
KS = 3
USE_MASK = True
USE_POOL = True


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        p = k // 2
        self.c1 = nn.Conv2d(c, c, k, padding=p)
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, k, padding=p)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        return F.relu(x + self.n2(self.c2(h)))


class FootHead(nn.Module):
    def __init__(self, c, fps, hid, pool):
        super().__init__()
        self.fps = fps
        self.pool = pool
        self.fconv = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in fps])
        if pool:
            self.proj = nn.Conv2d(c, hid, 1)
            self.mix = nn.ModuleList([nn.Conv2d(3 * hid, hid, 1) for _ in fps])
            self.hid = nn.ModuleList([nn.Conv2d(2 * hid, hid, 1) for _ in fps])
        else:
            self.hid = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in fps])

    def forward(self, u):
        p = self.proj(u) if self.pool else None
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            uq = F.pad(u, (0, dx - 1, 0, dy - 1))
            h = F.relu(self.fconv[i](uq))
            if self.pool:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
                h1 = F.relu(self.mix[i](torch.cat([mx, mn, av], 1)))
                h = torch.cat([h, h1], 1)
            h = F.relu(self.hid[i](h))
            res.append(self.outs[i](h))
        return torch.cat(res, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.Y = spec.Y
        self.X = spec.X
        self.n_place = spec.n_place
        cin = C + (self.T if USE_MASK else 0)
        self.s1 = nn.Conv2d(cin, 64, 1)
        self.s2 = nn.Conv2d(64, W, 1)
        self.n0 = nn.GroupNorm(8, W)
        self.blocks = nn.ModuleList([Block(W, KS) for _ in range(NBLK)])
        self.skip = nn.Conv2d(cin, SK, 1)
        U = W + SK
        self.head = FootHead(U, self.fps, HID, USE_POOL)
        self.tmix = nn.Parameter(torch.zeros(self.T, self.T))
        self.glob = nn.Linear(2 * U, 64)
        self.glob2 = nn.Linear(64, 64)
        self.dense = nn.Linear(64, spec.n_actions)
        self.value_head = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for m in self.head.outs:
            ortho(m, 0.01)
        ortho(self.dense, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        x = obs
        if USE_MASK:
            m = mask[:, :self.n_place].float().reshape(-1, self.T, self.Y, self.X)
            x = torch.cat([obs, m], 1)
        s = F.relu(self.skip(x))
        h = F.relu(self.n0(self.s2(F.relu(self.s1(x)))))
        for b in self.blocks:
            h = b(h)
        u = torch.cat([h, s], 1)
        maps = self.head(u)
        maps = maps + torch.einsum('tk,bkyx->btyx', self.tmix, maps)
        place = maps.flatten(1)
        g = F.relu(self.glob(torch.cat([u.mean((2, 3)), u.amax((2, 3))], 1)))
        g = F.relu(self.glob2(g))
        logits = torch.cat([place, place.new_zeros(place.shape[0], 1)], 1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
