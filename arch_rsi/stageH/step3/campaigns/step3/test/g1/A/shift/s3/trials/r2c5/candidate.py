import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
HID = 32
DCH = 4
GW = 64
DONE_SCALE = 0.1

RECIPE = {"lr": 0.0015, "ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = spec.n_types
        self.Y = spec.Y
        self.X = spec.X
        self.n_place = spec.n_place
        C = spec.in_channels + spec.n_types
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(C, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, COL, 1)), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, WIDTH)
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(WIDTH, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.dproj = init_layer(nn.Conv2d(WIDTH, DCH, 1))
        self.dense = init_layer(nn.Linear(DCH * spec.Y * spec.X, spec.n_place), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], 1))
        x = F.relu(self.gn1(self.c1(s)))
        f = F.relu(x + self.gn2(self.c2(x)))
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(ex(q))))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        place = torch.cat(res, 1) + self.type_term(g)[:, :, None, None]
        d = F.relu(self.dproj(f)).flatten(1)
        place = place.flatten(1) + self.dense(d)
        done = DONE_SCALE * self.done(g)
        logits = torch.cat([place, done], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
