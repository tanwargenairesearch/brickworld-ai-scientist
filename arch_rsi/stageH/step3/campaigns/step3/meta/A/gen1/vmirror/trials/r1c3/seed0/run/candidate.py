import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.003, "n_envs": 256, "rollout_len": 64, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.995, "gae_lambda": 0.99, "clip_eps": 0.3}

STEM = 64
COL = 48
WIDTH = 48
PROJ = 32
HID = 32
SKIP = 16


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def fp_stats(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for fp in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for fp in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = fp_stats(p, dy, dx)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.n_types = spec.n_types
        self.stem = nn.Sequential(nn.Conv2d(c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1))
        self.ln0 = CLN(COL)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.ln1 = CLN(WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ln2 = CLN(WIDTH)
        self.inter = PoolHead(WIDTH + SKIP, spec.footprints, PROJ, HID)
        self.anchor = nn.Conv2d(WIDTH + SKIP, 1, 1)
        self.glob = nn.Sequential(nn.Linear(2 * (WIDTH + SKIP), 64), nn.ReLU())
        self.type_head = nn.Linear(64, spec.n_types)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho(self, math.sqrt(2.0))
        for o in self.inter.outs:
            ortho(o, 0.01)
        ortho(self.anchor, 0.01)
        ortho(self.type_head, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.ln0(self.stem(obs)))
        k = F.relu(self.skip(obs))
        x = F.relu(self.ln1(self.c1(s)))
        f = F.relu(x + self.ln2(self.c2(x)))
        h = torch.cat([f, k], dim=1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        inter = self.inter(h)
        anc = self.anchor(h)
        typ = self.type_head(g)
        place = (inter + anc + typ[:, :, None, None]).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
