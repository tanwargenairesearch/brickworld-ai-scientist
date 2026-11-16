import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
K = 6
PROJ = 24
HID = 32
GW = 64
DONE_SCALE = 0.1

RECIPE = {"lr": 0.001, "ppo_epochs": 3, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def fpad(x, dy, dx):
    return F.pad(x, (0, dx - 1, 0, dy - 1))


def maxp(x, dy, dx):
    if dy * dx == 1:
        return x
    return F.max_pool2d(fpad(x, dy, dx), (dy, dx), stride=1)


def minp(x, dy, dx):
    if dy * dx == 1:
        return x
    return -F.max_pool2d(-fpad(x, dy, dx), (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.Z = spec.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(C, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, COL, 1)), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, WIDTH)
        fin = WIDTH + COL + C
        self.hgt = init_layer(nn.Conv2d(fin, spec.Z, 1), 1.0)
        self.cell = init_layer(nn.Conv2d(fin, spec.Z * K, 1), 1.0)
        self.proj = init_layer(nn.Conv2d(fin, PROJ, 1))
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(2 * K, HID, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH + COL, GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        s = self.stem(obs)
        x = F.relu(self.gn1(self.c1(s)))
        f = F.relu(x + self.gn2(self.c2(x)))
        z = torch.cat([f, s, obs], 1)
        u = torch.sigmoid(self.hgt(z))
        m = torch.sigmoid(self.cell(z))
        p = F.relu(self.proj(z))
        res = []
        for (dy, dx), ex, mix, out in zip(self.fps, self.exact, self.mix, self.outs):
            U = maxp(u, dy, dx)
            prev = torch.cat([torch.ones_like(U[:, :1]), U[:, :-1]], 1)
            ell = (prev - U).unsqueeze(2)
            mn = minp(m, dy, dx).reshape(B, self.Z, K, Y, X)
            mx = maxp(m, dy, dx).reshape(B, self.Z, K, Y, X)
            a = (ell * mn).sum(1)
            b = (ell * mx).sum(1)
            h = F.relu(ex(fpad(p, dy, dx)) + mix(torch.cat([a, b], 1)))
            res.append(out(h))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), s.amax((2, 3))], 1))
        place = torch.cat(res, 1) + self.type_term(g)[:, :, None, None]
        done = DONE_SCALE * self.done(g)
        logits = torch.cat([place.flatten(1), done], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
