import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 64, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

S1 = 64
W = 32
NB = 3
RAW = 16
HID = 32
DF = 16
GH = 128


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        g = math.sqrt(2.0)
        self.c1 = init_layer(nn.Conv2d(w, w, 3, padding=1), g)
        self.n1 = nn.GroupNorm(8, w)
        self.c2 = init_layer(nn.Conv2d(w, w, 1), g)
        self.n2 = nn.GroupNorm(8, w)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(x + h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c_in = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        g = math.sqrt(2.0)
        self.s1 = init_layer(nn.Conv2d(c_in, S1, 1), g)
        self.g1 = nn.GroupNorm(8, S1)
        self.s2 = init_layer(nn.Conv2d(S1, W, 1), g)
        self.g2 = nn.GroupNorm(8, W)
        self.blocks = nn.ModuleList([Block(W) for _ in range(NB)])
        self.raw = init_layer(nn.Conv2d(c_in, RAW, 1), 1.0)
        fd = W + RAW
        self.fconv = nn.ModuleList([init_layer(nn.Conv2d(fd, HID, (dy, dx)), g) for dy, dx in self.fps])
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(HID, HID, 1), g) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(fd, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.dfeat = init_layer(nn.Conv2d(fd, DF, 1), g)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * fd, GH), g), nn.ReLU())
        self.dmlp = nn.Sequential(init_layer(nn.Linear(2 * DF + GH, 32), g), nn.ReLU(),
                                  init_layer(nn.Linear(32, 1), 0.01))
        self.value = init_layer(nn.Linear(GH, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.g1(self.s1(obs)))
        s = F.relu(self.g2(self.s2(s)))
        for b in self.blocks:
            s = b(s)
        f = torch.cat([s, self.raw(obs)], dim=1)
        a = self.anchor(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(self.fconv[i](q))
            h = F.relu(self.mix[i](h))
            res.append(self.outs[i](h) + a + self.tbias[i])
        place = torch.cat(res, dim=1).flatten(1)
        d = F.relu(self.dfeat(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.dmlp(torch.cat([d.amax((2, 3)), d.mean((2, 3)), g], dim=1))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
