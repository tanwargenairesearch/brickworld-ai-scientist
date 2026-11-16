import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
RAW = 16
GW = 64
DONE_SCALE = 0.1

RECIPE = {"lr": 0.0005, "ppo_epochs": 8, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.15}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class PoolExactHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = init_layer(nn.Conv2d(c, proj, 1))
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(3 * proj, hid, 1)) for _ in self.fps])
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(proj, hid, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(hid, 1, 1), 0.01) for _ in self.fps])

    def forward(self, f):
        p = F.relu(self.proj(f))
        res = []
        for (dy, dx), mix, ex, out in zip(self.fps, self.mix, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], 1)) + ex(q))
            res.append(out(h))
        return torch.cat(res, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(C, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, COL, 1)), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, WIDTH)
        self.raw = init_layer(nn.Conv2d(C, RAW, 1))
        self.head = PoolExactHead(WIDTH + RAW, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH + COL, GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.gn1(self.c1(s)))
        f = F.relu(x + self.gn2(self.c2(x)))
        place = self.head(torch.cat([f, self.raw(obs)], 1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), s.amax((2, 3))], 1))
        place = place + self.type_term(g)[:, :, None, None]
        done = DONE_SCALE * self.done(g)
        logits = torch.cat([place.flatten(1), done], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
