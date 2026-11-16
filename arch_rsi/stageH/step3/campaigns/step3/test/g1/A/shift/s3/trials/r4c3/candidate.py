import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
GW = 64

RECIPE = {"lr": 0.0005, "ppo_epochs": 1, "n_envs": 64, "rollout_len": 8, "n_minibatches": 4, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.9, "clip_eps": 0.2}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.s1 = init_layer(nn.Conv2d(C, STEM, 1))
        self.s2 = init_layer(nn.Conv2d(STEM, COL, 1))
        self.gn0 = nn.GroupNorm(8, COL)
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, WIDTH)
        self.proj = init_layer(nn.Conv2d(COL + WIDTH, PROJ, 1))
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(PROJ, 1, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * (COL + WIDTH), GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.s1(obs))
        s = F.relu(self.gn0(self.s2(s)))
        x = F.relu(self.gn1(self.c1(s)))
        x = F.relu(x + self.gn2(self.c2(x)))
        feats = torch.cat([s, x], 1)
        p = F.relu(self.proj(feats))
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(ex(q))))
        g = self.glob(torch.cat([feats.mean((2, 3)), feats.amax((2, 3))], 1))
        place = torch.cat(res, 1) + self.anchor(p) + self.type_term(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
