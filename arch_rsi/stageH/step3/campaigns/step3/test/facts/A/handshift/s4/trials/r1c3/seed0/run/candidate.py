import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 3e-3, 'n_envs': 512, 'rollout_len': 8, 'n_minibatches': 8, 'ppo_epochs': 2, 'ent_coef': 0.005, 'gamma': 0.99, 'gae_lambda': 0.8, 'clip_eps': 0.2}

STEM = 48
WIDTH = 32
PROJ = 24
HID = 24


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
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
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.s1 = nn.Conv2d(spec.in_channels, STEM, 1)
        self.n1 = nn.GroupNorm(4, STEM)
        self.s2 = nn.Conv2d(STEM, WIDTH, 1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n3 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n4 = nn.GroupNorm(4, WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Linear(2 * WIDTH, 64)
        self.gn = nn.LayerNorm(64)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2.0))
        for o in self.head.outs:
            ortho(o, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.n1(self.s1(obs)))
        s = F.relu(self.n2(self.s2(s)))
        x = F.relu(self.n3(self.c1(s)) + s)
        f = F.relu(self.n4(self.c2(x)) + x)
        place = self.head(f)
        g = F.relu(self.gn(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
