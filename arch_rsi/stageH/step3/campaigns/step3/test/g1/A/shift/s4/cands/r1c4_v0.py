import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1e-3, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64,
          "ppo_epochs": 4, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

CELL = 96
COL = 64
WIDTH = 48
SKIP = 16
PROJ = 32
HID = 48
GLOB = 64
GROUPS = 8


def init_layer(m, gain=1.4142135623730951):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class PoolHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1))
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(3 * PROJ, HID, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.kh = max(a for a, b in self.fps)
        self.kw = max(b for a, b in self.fps)
        self.dense = init_layer(nn.Conv2d(c, len(self.fps), (self.kh, self.kw)), 0.01)
        self.anchor = init_layer(nn.Conv2d(c, 1, 1), 0.01)

    def forward(self, f, tbias):
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
        inter = torch.cat(res, dim=1)
        dense = self.dense(F.pad(f, (0, self.kw - 1, 0, self.kh - 1)))
        place = inter + dense + self.anchor(f) + tbias[:, :, None, None]
        return place.flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.m1 = init_layer(nn.Conv2d(cin, CELL, 1))
        self.g1 = nn.GroupNorm(GROUPS, CELL)
        self.m2 = init_layer(nn.Conv2d(CELL, CELL, 1))
        self.g2 = nn.GroupNorm(GROUPS, CELL)
        self.m3 = init_layer(nn.Conv2d(CELL, COL, 1))
        self.g3 = nn.GroupNorm(GROUPS, COL)
        self.c0 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.g4 = nn.GroupNorm(GROUPS, WIDTH)
        self.skip = init_layer(nn.Conv2d(cin, SKIP, 1))
        fc = WIDTH + COL + SKIP
        self.head = PoolHead(fc, spec.footprints)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * fc, GLOB)), nn.ReLU(),
                                  init_layer(nn.Linear(GLOB, GLOB)), nn.ReLU())
        self.type_bias = init_layer(nn.Linear(GLOB, spec.n_types), 0.01)
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.vhid = nn.Sequential(init_layer(nn.Linear(2 * fc, GLOB)), nn.ReLU())
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        h = F.relu(self.g1(self.m1(obs)))
        h = h + F.relu(self.g2(self.m2(h)))
        c = F.relu(self.g3(self.m3(h)))
        x = F.relu(self.g4(self.c0(c)))
        r = F.relu(self.skip(obs))
        f = torch.cat([x, c, r], dim=1)
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        place = self.head(f, self.type_bias(g))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        value = self.value_head(self.vhid(pooled)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
