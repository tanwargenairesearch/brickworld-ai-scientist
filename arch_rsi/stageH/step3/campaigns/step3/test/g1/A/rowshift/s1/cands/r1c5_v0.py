import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
DET = 16
PROJ = 32
HID = 40
GLOB = 64

RECIPE = {'ppo_epochs': 2, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 32, 'lr': 0.001, 'gamma': 0.97, 'gae_lambda': 0.95, 'ent_coef': 0.01}


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
        nt = len(self.fps)
        self.stem = nn.Sequential(
            nn.Conv2d(c, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.GroupNorm(8, COL), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.det = nn.Conv2d(c, DET, 1)
        self.proj = nn.Conv2d(WIDTH + SKIP, PROJ, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + DET, GLOB), nn.ReLU())
        self.gproj = nn.Linear(GLOB, HID)
        self.mix = nn.Conv2d(3 * PROJ, HID, 1)
        self.temb = nn.Parameter(torch.zeros(nt, HID))
        self.h2 = nn.Conv2d(HID, HID, 1)
        self.out = nn.Conv2d(HID, 1, 1)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        init_layers(self)
        nn.init.normal_(self.temb, std=0.1)
        init_out(self.gproj, 0.1)
        init_out(self.out, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        p = self.proj(torch.cat([f, self.skip(obs)], dim=1))
        d = F.relu(self.det(obs)).amax(dim=(2, 3))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), d], dim=1))
        gb = self.gproj(g).view(g.shape[0], -1, 1, 1)
        res = []
        i = 0
        for (dy, dx) in self.fps:
            h = F.relu(self.mix(fp_pool(p, dy, dx)) + self.temb[i].view(1, -1, 1, 1) + gb)
            h = F.relu(self.h2(h))
            res.append(self.out(h))
            i += 1
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
