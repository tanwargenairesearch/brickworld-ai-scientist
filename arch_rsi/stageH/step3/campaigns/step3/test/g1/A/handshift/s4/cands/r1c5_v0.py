import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2}

COLW = 48
SKIP = 24
SKIP_DEEP = False
WIDTH = 48
NBLOCKS = 3
USE_TRUNK = True
PROJ = 24
HID = 32
MIX2 = False
DENSE3 = True
FLATD = 0
GLOB = 64
GROUPS = 8


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.nt = nt
        cin = int(spec.in_channels)
        g = math.sqrt(2.0)
        self.col1 = ortho(nn.Conv2d(cin, COLW, 1), g)
        self.col2 = ortho(nn.Conv2d(COLW, COLW, 1), g)
        if SKIP_DEEP:
            self.skip = nn.Sequential(ortho(nn.Conv2d(cin, COLW, 1), g), nn.ReLU(), ortho(nn.Conv2d(COLW, SKIP, 1), g), nn.ReLU())
        else:
            self.skip = nn.Sequential(ortho(nn.Conv2d(cin, SKIP, 1), g), nn.ReLU())
        if USE_TRUNK:
            self.c0 = ortho(nn.Conv2d(COLW, WIDTH, 3, padding=1), g)
            self.n0 = nn.GroupNorm(GROUPS, WIDTH)
            self.blocks = nn.ModuleList([ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g) for _ in range(NBLOCKS)])
            self.norms = nn.ModuleList([nn.GroupNorm(GROUPS, WIDTH) for _ in range(NBLOCKS)])
            fw = WIDTH
        else:
            fw = COLW
        self.proj = ortho(nn.Conv2d(fw, PROJ, 1), g)
        P = PROJ + SKIP
        self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * P, HID, 1), g) for _ in self.fps])
        if MIX2:
            self.mix2 = nn.ModuleList([ortho(nn.Conv2d(HID, HID, 1), g) for _ in self.fps])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = ortho(nn.Conv2d(P, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(1, nt, 1, 1))
        if DENSE3:
            self.dense3 = ortho(nn.Conv2d(P, nt, 3, padding=1), 0.01)
        if FLATD > 0:
            self.flat_in = ortho(nn.Conv2d(P, FLATD, 1), g)
            self.flat_out = ortho(nn.Linear(FLATD * self.Y * self.X, nt * self.Y * self.X), 0.01)
        gin = 2 * fw + 2 * P
        self.glob = nn.Sequential(ortho(nn.Linear(gin, GLOB), g), nn.ReLU())
        self.ctx = ortho(nn.Linear(GLOB, nt * HID), 1.0)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        h = F.relu(self.col1(obs))
        h = F.relu(self.col2(h))
        s = self.skip(obs)
        if USE_TRUNK:
            x = F.relu(self.n0(self.c0(h)))
            for conv, nrm in zip(self.blocks, self.norms):
                x = F.relu(x + nrm(conv(x)))
        else:
            x = h
        p = torch.cat([F.relu(self.proj(x)), s], dim=1)
        gfeat = torch.cat([x.mean((2, 3)), x.amax((2, 3)), p.mean((2, 3)), p.amax((2, 3))], dim=1)
        g = self.glob(gfeat)
        ctx = self.ctx(g).view(B, self.nt, HID, 1, 1)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hh = F.relu(self.mix[i](torch.cat([mx, mn, av], 1)) + ctx[:, i])
            if MIX2:
                hh = F.relu(self.mix2[i](hh))
            res.append(self.outs[i](hh))
        place = torch.cat(res, 1) + self.anchor(p) + self.type_bias
        if DENSE3:
            place = place + self.dense3(p)
        place = place.flatten(1)
        if FLATD > 0:
            fl = F.relu(self.flat_in(p)).flatten(1)
            place = place + self.flat_out(fl)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
