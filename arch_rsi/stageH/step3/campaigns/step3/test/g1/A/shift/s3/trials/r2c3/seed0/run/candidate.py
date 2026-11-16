import math
import torch
import torch.nn as nn
import torch.nn.functional as F

COLW = 96
FEAT = 48
CTX = 40
RAW = 16
HID = 32
GW = 64
DONE_SCALE = 0.2

RECIPE = {"lr": 0.001, "ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(c, hid, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(hid, 1, 1), 0.01) for _ in self.fps])

    def forward(self, p):
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(ex(q))))
        return torch.cat(res, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.col1 = init_layer(nn.Conv2d(C, COLW, 1))
        self.gn0 = nn.GroupNorm(8, COLW)
        self.col2 = init_layer(nn.Conv2d(COLW, FEAT, 1))
        self.c1 = init_layer(nn.Conv2d(FEAT, CTX, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, CTX)
        self.c2 = init_layer(nn.Conv2d(CTX, CTX, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, CTX)
        self.raw = init_layer(nn.Conv2d(C, RAW, 1))
        self.proj = init_layer(nn.Conv2d(FEAT + CTX + RAW, FEAT, 1))
        self.head = FootHead(FEAT, spec.footprints, HID)
        self.anchor = init_layer(nn.Conv2d(FEAT, 1, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * (FEAT + CTX), GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.dense = init_layer(nn.Linear(GW, spec.n_place), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        c = F.relu(self.gn0(self.col1(obs)))
        c = F.relu(self.col2(c))
        x = F.relu(self.gn1(self.c1(c)))
        x = F.relu(x + self.gn2(self.c2(x)))
        r = self.raw(obs)
        p = F.relu(self.proj(torch.cat([c, x, r], 1)))
        feats = torch.cat([c, x], 1)
        g = self.glob(torch.cat([feats.mean((2, 3)), feats.amax((2, 3))], 1))
        place = self.head(p) + self.anchor(p) + self.type_term(g)[:, :, None, None]
        place = place.flatten(1) + self.dense(g)
        done = -F.softplus(DONE_SCALE * self.done(g))
        logits = torch.cat([place, done], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
