import math
import torch
import torch.nn as nn
import torch.nn.functional as F

COL1 = 64
COL2 = 48
W = 40
RAW = 16
PROJ = 32
HID = 24
HH = 24
GW = 64

RECIPE = {"lr": 0.002, "ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.05, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.3}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col1 = init_layer(nn.Conv2d(C, COL1, 1))
        self.col2 = init_layer(nn.Conv2d(COL1, COL2, 1))
        self.gn0 = nn.GroupNorm(8, COL2)
        self.c1 = init_layer(nn.Conv2d(COL2, W, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, W)
        self.c2 = init_layer(nn.Conv2d(W, W, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, W)
        self.raw = init_layer(nn.Conv2d(C, RAW, 1))
        self.proj = init_layer(nn.Conv2d(W + RAW + COL2, PROJ, 1))
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.h3 = init_layer(nn.Conv2d(PROJ, HH, 3, padding=1))
        self.hout = init_layer(nn.Conv2d(HH, spec.n_types, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * W + COL2, GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        c = F.relu(self.col1(obs))
        c = F.relu(self.gn0(self.col2(c)))
        x = F.relu(self.gn1(self.c1(c)))
        f = F.relu(x + self.gn2(self.c2(x)))
        p = F.relu(self.proj(torch.cat([f, self.raw(obs), c], 1)))
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(ex(q))))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), c.amax((2, 3))], 1))
        place = torch.cat(res, 1) + self.hout(F.relu(self.h3(p))) + self.type_term(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
