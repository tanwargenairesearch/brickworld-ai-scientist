import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 3, 'gamma': 0.995, 'gae_lambda': 0.95, 'ent_coef': 0.002}

STEM = 64
COL = 48
WIDTH = 40
SK = 16
PROJ = 32
HID = 32
G = 64


def ortho(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    B, C, Y, X = obs.shape
    yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([obs, yy, xx], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels + 2
        T = len(self.fps)
        self.s1 = ortho(nn.Conv2d(C, STEM, 1))
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1))
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), 1.0)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(C, SK, 1))
        D = WIDTH + SK
        self.proj = ortho(nn.Conv2d(D, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * PROJ, HID, 1)) for t in range(T)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for t in range(T)])
        self.anchor = ortho(nn.Conv2d(D, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.glob = nn.Sequential(ortho(nn.Linear(2 * D, G)), nn.ReLU(), ortho(nn.Linear(G, G)), nn.ReLU())
        self.done = ortho(nn.Linear(G, 1), 0.01)
        self.vnet = nn.Sequential(ortho(nn.Linear(2 * D, G)), nn.ReLU(), ortho(nn.Linear(G, 1), 1.0))

    def forward(self, obs, mask):
        o = add_coords(obs)
        s = F.relu(self.s2(F.relu(self.s1(o))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(self.n2(x + self.c2(x)))
        k = F.relu(self.skip(o))
        h = torch.cat([f, k], dim=1)
        p = self.proj(h)
        a = self.anchor(h)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            z = F.relu(self.mix[i](torch.cat([mx, mn, av], dim=1)))
            res.append(self.outs[i](z) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.vnet(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
