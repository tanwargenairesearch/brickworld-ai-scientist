import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 32,
          "gamma": 0.99, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.005}

STEM = 64
W = 40
RAW = 24
PH = 32
PP = 24
PM = 24
GH = 48


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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        c = spec.in_channels
        self.stem1 = nn.Conv2d(c, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, W, 1)
        self.gn0 = nn.GroupNorm(8, W)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, W)
        self.raw = nn.Conv2d(c, RAW, 1)
        feat = W + RAW
        self.foot = nn.ModuleList([nn.Conv2d(feat, PH, (dy, dx)) for dy, dx in self.fps])
        self.pproj = nn.Conv2d(feat, PP, 1)
        self.pmix = nn.ModuleList([nn.Conv2d(3 * PP, PM, 1) for _ in self.fps])
        self.comb = nn.ModuleList([nn.Conv2d(PH + PM, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(feat, 1, 1)
        self.gproj = nn.Conv2d(feat, GH, 1)
        self.glob = nn.Linear(2 * GH, 64)
        self.type_bias = nn.Linear(64, len(self.fps))
        self.done = nn.Linear(64, 1)
        self.value = nn.Linear(64, 1)
        init_all(self)
        for o in self.comb:
            init_out(o, 0.01)
        init_out(self.anchor, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done, 0.01)
        init_out(self.value, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.gn0(self.stem2(F.relu(self.stem1(obs)))))
        x = F.relu(s + self.gn1(self.c1(s)))
        x = F.relu(x + self.gn2(self.c2(x)))
        f = torch.cat([x, F.relu(self.raw(obs))], dim=1)
        p = self.pproj(f)
        gp = F.relu(self.gproj(f))
        g = F.relu(self.glob(torch.cat([gp.amax((2, 3)), gp.mean((2, 3))], dim=1)))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h1 = F.relu(self.foot[i](q))
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                pq = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(pq, (dy, dx), stride=1)
                mn = -F.max_pool2d(-pq, (dy, dx), stride=1)
                av = F.avg_pool2d(pq, (dy, dx), stride=1)
            h2 = F.relu(self.pmix[i](torch.cat([mx, mn, av], dim=1)))
            res.append(self.comb[i](torch.cat([h1, h2], dim=1)))
        place = torch.cat(res, dim=1) + self.anchor(f) + self.type_bias(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
