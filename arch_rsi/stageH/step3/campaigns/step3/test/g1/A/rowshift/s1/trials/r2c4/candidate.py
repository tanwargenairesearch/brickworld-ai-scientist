import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 48
NBLK = 6
SKIP = 16
DET = 16
PROJ = 32
HID = 32
GLOB = 64

RECIPE = {'ppo_epochs': 1, 'n_envs': 1024, 'rollout_len': 8, 'n_minibatches': 64, 'lr': 0.0005, 'gamma': 0.995, 'gae_lambda': 0.95, 'clip_eps': 0.1, 'ent_coef': 0.002}


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
        self.stem = nn.Sequential(
            nn.Conv2d(c, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.GroupNorm(8, WIDTH), nn.ReLU())
        dws = []
        pws = []
        gns = []
        for k in range(NBLK):
            dil = 1 + (k % 2)
            dws.append(nn.Conv2d(WIDTH, WIDTH, 3, padding=dil, dilation=dil, groups=WIDTH))
            pws.append(nn.Conv2d(WIDTH, WIDTH, 1))
            gns.append(nn.GroupNorm(8, WIDTH))
        self.dws = nn.ModuleList(dws)
        self.pws = nn.ModuleList(pws)
        self.gns = nn.ModuleList(gns)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.proj = nn.Conv2d(WIDTH + SKIP, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.det = nn.Conv2d(c, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + DET, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        init_layers(self)
        for o in self.outs:
            init_out(o, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for dw, pw, gn in zip(self.dws, self.pws, self.gns):
            y = pw(F.relu(dw(x)))
            x = F.relu(x + gn(y))
        f = x
        p = self.proj(torch.cat([f, self.skip(obs)], dim=1))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            h = F.relu(mix(fp_pool(p, dy, dx)))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        d = F.relu(self.det(obs)).amax(dim=(2, 3))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), d], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
