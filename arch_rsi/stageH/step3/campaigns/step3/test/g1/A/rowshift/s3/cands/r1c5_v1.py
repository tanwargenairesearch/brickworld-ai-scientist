import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 48, "n_minibatches": 24, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.995, "gae_lambda": 0.95, "clip_eps": 0.3}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
RG = 1.4142135623730951


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def norm(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = init_(nn.Conv2d(c, proj, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * proj, hid, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(hid, 1, 1), 0.01) for _ in self.fps])

    def forward(self, f):
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
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem1 = init_(nn.Conv2d(spec.in_channels, STEM, 1), RG)
        self.stem2 = init_(nn.Conv2d(STEM, COL, 1), RG)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), RG)
        self.n1 = norm(WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), RG)
        self.n2 = norm(WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.gnorm = nn.LayerNorm(2 * WIDTH)
        self.glob = init_(nn.Linear(2 * WIDTH, 64), RG)
        self.done_head = init_(nn.Linear(64, 1), 0.01)
        self.value_head = init_(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        place = self.head(f)
        g = F.relu(self.glob(self.gnorm(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
