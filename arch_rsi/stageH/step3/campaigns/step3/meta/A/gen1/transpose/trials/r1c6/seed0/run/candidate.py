import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "ppo_epochs": 2, "rollout_len": 32, "n_minibatches": 32,
          "clip_eps": 0.3, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.005}

W = 40
SK = 32
PROJ = 32
HID = 32


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(x + h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.s1 = nn.Conv2d(C, 64, 1)
        self.s2 = nn.Conv2d(64, W, 1)
        self.n0 = nn.GroupNorm(8, W)
        self.block = Block(W)
        self.skip = nn.Conv2d(C, SK, 1)
        U = W + SK
        self.proj = nn.Conv2d(U, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.fconv = nn.ModuleList([nn.Conv2d(U, HID, (dy, dx)) for dy, dx in self.fps])
        self.hid = nn.ModuleList([nn.Conv2d(2 * HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(U, 1, 1)
        self.glob = nn.Linear(2 * W + 2 * SK, 64)
        self.type_bias = nn.Linear(64, len(self.fps))
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for m in self.outs:
            ortho(m, 0.01)
        ortho(self.anchor, 0.01)
        ortho(self.type_bias, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.skip(obs))
        h = F.relu(self.n0(self.s2(F.relu(self.s1(obs)))))
        f = self.block(h)
        u = torch.cat([f, s], dim=1)
        p = self.proj(u)
        a = self.anchor(u)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                                         s.mean((2, 3)), s.amax((2, 3))], dim=1)))
        tb = self.type_bias(g)
        res = []
        for i in range(len(self.fps)):
            dy, dx = self.fps[i]
            if dy * dx == 1:
                mx, mn, av = p, p, p
                uq = u
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
                uq = F.pad(u, (0, dx - 1, 0, dy - 1))
            h1 = F.relu(self.mix[i](torch.cat([mx, mn, av], dim=1)))
            h2 = F.relu(self.fconv[i](uq))
            hh = F.relu(self.hid[i](torch.cat([h1, h2], dim=1)))
            res.append(self.outs[i](hh) + a + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
