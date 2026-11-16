import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CELL1 = 128
CELL2 = 64
CONV = 32
SKIP = 16
DET = 16
PROJ = 32
HID = 48
GLOB = 64

RECIPE = {'ppo_epochs': 1, 'n_envs': 512, 'rollout_len': 16, 'n_minibatches': 64, 'lr': 0.002, 'gamma': 0.9, 'gae_lambda': 0.8, 'ent_coef': 0.003}


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.cell = nn.Sequential(
            nn.Conv2d(c, CELL1, 1), nn.ReLU(),
            nn.Conv2d(CELL1, CELL2, 1), nn.GroupNorm(8, CELL2), nn.ReLU())
        self.res = nn.Conv2d(CELL2, CELL2, 1)
        self.nres = nn.GroupNorm(8, CELL2)
        self.conv = nn.Conv2d(CELL2, CONV, 3, padding=1)
        self.nconv = nn.GroupNorm(8, CONV)
        self.skip = nn.Conv2d(c, SKIP, 1)
        fch = CELL2 + CONV
        self.proj = nn.Conv2d(fch + SKIP, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.h2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.det = nn.Conv2d(c, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * fch + DET, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        init_layers(self)
        for o in self.outs:
            init_out(o, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        a = self.cell(obs)
        a = F.relu(a + self.nres(self.res(a)))
        b = F.relu(self.nconv(self.conv(a)))
        f = torch.cat([a, b], dim=1)
        p = self.proj(torch.cat([f, self.skip(obs)], dim=1))
        res = []
        for (dy, dx), mix, h2, out in zip(self.fps, self.mix, self.h2, self.outs):
            h = F.relu(mix(fp_pool(p, dy, dx)))
            h = F.relu(h2(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        d = F.relu(self.det(obs)).amax(dim=(2, 3))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), d], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
