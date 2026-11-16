import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
W = 40
PROJ = 24
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.hgt = nn.Conv2d(W, self.Z + 1, 1)
        self.ok = nn.Conv2d(W, self.Z, 1)
        self.scale = nn.Parameter(torch.ones(len(self.fps)))
        self.proj = nn.Conv2d(W, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, obs, mask):
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        cdf = torch.cumsum(F.softmax(self.hgt(f), dim=1), dim=1).clamp(1e-6, 1.0)
        lcdf = torch.log(cdf)
        lok = F.logsigmoid(self.ok(f))
        p = self.proj(f)
        res = []
        for k, (dy, dx) in enumerate(self.fps):
            n = float(dy * dx)
            if dy * dx == 1:
                lc, lo = lcdf, lok
                mx, mn, av = p, p, p
            else:
                pad = (0, dx - 1, 0, dy - 1)
                lc = F.avg_pool2d(F.pad(lcdf, pad), (dy, dx), stride=1) * n
                lo = F.avg_pool2d(F.pad(lok, pad), (dy, dx), stride=1) * n
                q = F.pad(p, pad)
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            cf = lc.exp()
            prev = torch.cat([torch.zeros_like(cf[:, :1]), cf[:, :Z - 1]], dim=1)
            pmf = (cf[:, :Z] - prev).clamp(min=0.0)
            score = (pmf * lo.exp()).sum(1, keepdim=True)
            struct = torch.log(score + 1e-6)
            resid = self.outs[k](F.relu(self.mix[k](torch.cat([mx, mn, av], dim=1))))
            res.append(resid + self.scale[k] * struct)
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
