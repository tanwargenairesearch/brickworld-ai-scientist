import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 32, "ppo_epochs": 3, "n_minibatches": 32, "ent_coef": 0.005, "lr": 0.001}

STEM = 64
COL = 48
WIDTH = 40
RAW = 16
PROJ = 32
HID = 32
G = math.sqrt(2.0)


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def pool_stats(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        T = len(self.fps)
        self.s1 = init_(nn.Conv2d(C, STEM, 1), G)
        self.s2 = init_(nn.Conv2d(STEM, COL, 1), G)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), G)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), G)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.raw = init_(nn.Conv2d(C, RAW, 1), G)
        self.proj = init_(nn.Conv2d(WIDTH + RAW, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), G) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(WIDTH + RAW, 1, 1), 0.01)
        self.glob = init_(nn.Linear(2 * WIDTH, 64), G)
        self.type_bias = init_(nn.Linear(64, T), 0.01)
        self.done_head = init_(nn.Linear(64, 1), 0.01)
        self.value_head = init_(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        fr = torch.cat([f, F.relu(self.raw(obs))], 1)
        p = self.proj(fr)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        tb = self.type_bias(g)
        a = self.anchor(fr)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = pool_stats(p, dy, dx)
            h = F.relu(self.mix[i](torch.cat([mx, mn, av], 1)))
            res.append(self.outs[i](h) + a + tb[:, i].view(B, 1, 1, 1))
        place = torch.cat(res, 1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
