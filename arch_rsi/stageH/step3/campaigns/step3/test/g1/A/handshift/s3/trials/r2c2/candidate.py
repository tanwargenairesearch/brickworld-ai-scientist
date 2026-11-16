import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.25}

H1 = 48
COL = 48
W = 40
SKIP = 16
PROJ = 32
HID = 24
G = 128
DSK = 16


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def fp_pool(p, dy, dx):
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
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(c, H1, 1), nn.GroupNorm(8, H1), nn.ReLU(),
            nn.Conv2d(H1, H1, 1), nn.GroupNorm(8, H1), nn.ReLU(),
            nn.Conv2d(H1, COL, 1), nn.GroupNorm(8, COL), nn.ReLU())
        self.c1 = nn.Conv2d(COL, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(8, W)
        self.c3 = nn.Conv2d(W, W, 3, padding=2, dilation=2)
        self.n3 = nn.GroupNorm(8, W)
        self.skip = nn.Conv2d(c, SKIP, 1)
        hin = W + COL + SKIP
        self.proj = nn.Conv2d(hin, PROJ, 1)
        self.fpc = nn.ModuleList([nn.Conv2d(PROJ, PROJ, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(4 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(hin, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.dsk = nn.Conv2d(c, DSK, 1)
        self.glob = nn.Sequential(nn.Linear(2 * hin + 2 * DSK, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        ortho(self, math.sqrt(2))
        for o in self.outs:
            ortho(o, 0.01)
        ortho(self.anchor, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        col = self.col(obs)
        x = F.relu(self.n1(self.c1(col)))
        x = F.relu(x + self.n2(self.c2(x)))
        x = F.relu(x + self.n3(self.c3(x)))
        h = torch.cat([x, col, self.skip(obs)], dim=1)
        p = self.proj(h)
        res = []
        for (dy, dx), fc, mix, out in zip(self.fps, self.fpc, self.mix, self.outs):
            mx, mn, av = fp_pool(p, dy, dx)
            cv = fc(F.pad(p, (0, dx - 1, 0, dy - 1)))
            res.append(out(F.relu(mix(torch.cat([mx, mn, av, cv], dim=1)))))
        place = torch.cat(res, dim=1) + self.anchor(h) + self.tbias.view(1, -1, 1, 1)
        place = place.flatten(1)
        d = F.relu(self.dsk(obs))
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), d.mean((2, 3)), d.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
