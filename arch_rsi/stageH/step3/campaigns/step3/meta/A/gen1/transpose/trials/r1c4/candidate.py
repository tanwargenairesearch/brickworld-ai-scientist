import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 3, "rollout_len": 32, "n_minibatches": 32,
          "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

W = 40
D = 64
SK = 32
PROJ = 32
HID = 32


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.s1 = nn.Conv2d(C, 64, 1)
        self.s2 = nn.Conv2d(64, W, 1)
        self.n0 = nn.GroupNorm(8, W)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.down = nn.Conv2d(W, D, 3, stride=2, padding=1)
        self.nd = nn.GroupNorm(8, D)
        self.mid = nn.Conv2d(D, D, 3, padding=1)
        self.nm = nn.GroupNorm(8, D)
        self.up = nn.Conv2d(D, W, 1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(8, W)
        self.skip = nn.Conv2d(C, SK, 1)
        self.proj = nn.Conv2d(W + SK, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Linear(2 * W + 2 * D + SK, 64)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for m in self.outs:
            ortho(m, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.skip(obs))
        h = F.relu(self.n0(self.s2(F.relu(self.s1(obs)))))
        x = F.relu(h + self.n1(self.c1(h)))
        d = F.relu(self.nd(self.down(x)))
        d = F.relu(d + self.nm(self.mid(d)))
        upv = F.interpolate(d, size=(x.shape[2], x.shape[3]), mode='nearest')
        x = F.relu(x + self.up(upv))
        f = F.relu(x + self.n2(self.c2(x)))
        u = torch.cat([f, s], dim=1)
        p = self.proj(u)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        place = torch.cat(res, dim=1).flatten(1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                                         d.mean((2, 3)), d.amax((2, 3)),
                                         s.amax((2, 3))], dim=1)))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
