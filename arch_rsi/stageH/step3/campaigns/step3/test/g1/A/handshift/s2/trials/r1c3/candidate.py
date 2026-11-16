import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 32, 'ppo_epochs': 2, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.01}

W = 48
SK = 24
PROJ = 24
HID = 32
G = 64


def ortho(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


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
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        T = len(self.fps)
        self.stem = ortho(nn.Conv2d(C, W, 1))
        self.conv = ortho(nn.Conv2d(W, W, 3, padding=1), 1.0)
        self.norm = nn.GroupNorm(4, W)
        self.skip = ortho(nn.Conv2d(C, SK, 1))
        self.proj = ortho(nn.Conv2d(W + SK, PROJ, 1), 1.0)
        self.ga = nn.ModuleList([ortho(nn.Conv2d(3 * PROJ, HID, 1), 1.0) for t in range(T)])
        self.gb = nn.ModuleList([ortho(nn.Conv2d(3 * PROJ, HID, 1), 1.0) for t in range(T)])
        self.gc = nn.ModuleList([ortho(nn.Conv2d(3 * PROJ, HID, 1)) for t in range(T)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for t in range(T)])
        self.anchor = ortho(nn.Conv2d(W + SK, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.glob = nn.Sequential(ortho(nn.Linear(2 * (W + SK), G)), nn.ReLU(), ortho(nn.Linear(G, G)), nn.ReLU())
        self.done = ortho(nn.Linear(G, 1), 0.01)
        self.vnet = nn.Sequential(ortho(nn.Linear(2 * (W + SK), G)), nn.ReLU(), ortho(nn.Linear(G, 1), 1.0))

    def forward(self, obs, mask):
        s = F.relu(self.stem(obs))
        f = F.relu(self.norm(s + self.conv(s)))
        k = F.relu(self.skip(obs))
        h = torch.cat([f, k], dim=1)
        p = self.proj(h)
        a = self.anchor(h)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = fp_pool(p, dy, dx)
            u = torch.cat([mx, mn, av], dim=1)
            z = self.ga[i](u) * self.gb[i](u) + F.relu(self.gc[i](u))
            res.append(self.outs[i](z) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.vnet(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
