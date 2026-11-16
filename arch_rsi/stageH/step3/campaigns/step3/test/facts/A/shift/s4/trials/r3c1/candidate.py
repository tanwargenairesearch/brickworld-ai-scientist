import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"lr": 1e-3, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.01}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
VMUL = 0.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = ortho(nn.Conv2d(c, proj, 1), math.sqrt(2))
        self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * proj, hid, 1), math.sqrt(2)) for _ in self.fps])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(hid, 1, 1), 0.01) for _ in self.fps])

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
        g2 = math.sqrt(2)
        self.s1 = ortho(nn.Conv2d(spec.in_channels, STEM, 1), g2)
        self.n1 = nn.GroupNorm(8, STEM)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), g2)
        self.n2 = nn.GroupNorm(8, COL)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), g2)
        self.n3 = nn.GroupNorm(8, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g2)
        self.n4 = nn.GroupNorm(8, WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = ortho(nn.Linear(2 * WIDTH, 64), g2)
        self.gn = nn.LayerNorm(64)
        self.done_head = ortho(nn.Linear(64, 1), 0.01)
        self.value_head = ortho(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.n1(self.s1(obs)))
        s = F.relu(self.n2(self.s2(s)))
        x = F.relu(self.n3(self.c1(s)))
        f = F.relu(x + self.n4(self.c2(x)))
        place = self.head(f)
        g = F.relu(self.gn(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        v = VMUL * self.value_head(g).squeeze(1)
        return logits, v


def build_policy(spec):
    return Net(spec)
