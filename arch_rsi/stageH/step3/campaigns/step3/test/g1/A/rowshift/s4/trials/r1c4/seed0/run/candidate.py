import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 64, "n_minibatches": 8,
          "lr": 0.003, "clip_eps": 0.3, "gamma": 0.995, "gae_lambda": 0.99, "ent_coef": 0.01}

STEM = 64
COL = 48
WIDTH = 40
RAW = 24
PROJ = 32
HID = 32


def init_all(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

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
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem1 = nn.Conv2d(c, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.gn0 = nn.GroupNorm(8, COL)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, WIDTH)
        self.raw = nn.Conv2d(c, RAW, 1)
        feat = WIDTH + RAW
        self.head = PoolHead(feat, spec.footprints, PROJ, HID)
        self.anchor = nn.Conv2d(feat, 1, 1)
        self.glob = nn.Linear(2 * feat, 64)
        self.type_bias = nn.Linear(64, len(spec.footprints))
        self.done = nn.Linear(64, 1)
        self.value = nn.Linear(64, 1)
        init_all(self)
        for o in self.head.outs:
            init_out(o, 0.01)
        init_out(self.anchor, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done, 0.01)
        init_out(self.value, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.gn0(self.stem2(F.relu(self.stem1(obs)))))
        x = F.relu(self.gn1(self.c1(s)))
        x = F.relu(x + self.gn2(self.c2(x)))
        f = torch.cat([x, F.relu(self.raw(obs))], dim=1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        place = self.head(f) + self.anchor(f) + self.type_bias(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
