import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 3, 'gamma': 0.9, 'gae_lambda': 0.8, 'ent_coef': 0.003}

H = 80
SK = 24
K = 32
CT = 24
HID = 40
G = 64


def ortho(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def cln(norm, x):
    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def mm_pool(s, dy, dx):
    if dy * dx == 1:
        return s, s
    q = F.pad(s, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    return mn, mx


def av_pool(s, dy, dx):
    if dy * dx == 1:
        return s
    q = F.pad(s, (0, dx - 1, 0, dy - 1))
    return F.avg_pool2d(q, (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        T = len(self.fps)
        self.e1 = ortho(nn.Conv2d(C, H, 1))
        self.n1 = nn.LayerNorm(H)
        self.e2 = ortho(nn.Conv2d(H, H, 1), 1.0)
        self.n2 = nn.LayerNorm(H)
        self.e3 = ortho(nn.Conv2d(H, H, 1), 1.0)
        self.skip = ortho(nn.Conv2d(C, SK, 1))
        self.sf = ortho(nn.Conv2d(H + SK, K, 1), 1.0)
        self.ctx = ortho(nn.Conv2d(H, CT, 3, padding=1))
        self.mix = nn.ModuleList([ortho(nn.Conv2d(2 * K + 2 * CT, HID, 1)) for t in range(T)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for t in range(T)])
        self.anchor = ortho(nn.Conv2d(H + SK, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(T))
        P = 2 * K + 2 * (H + SK)
        self.glob = nn.Sequential(ortho(nn.Linear(P, G)), nn.ReLU(), ortho(nn.Linear(G, G)), nn.ReLU())
        self.done = ortho(nn.Linear(G, 1), 0.01)
        self.vnet = nn.Sequential(ortho(nn.Linear(P, G)), nn.ReLU(), ortho(nn.Linear(G, 1), 1.0))

    def forward(self, obs, mask):
        x = F.relu(self.e1(obs))
        x = x + self.e2(F.relu(cln(self.n1, x)))
        x = F.relu(x + self.e3(F.relu(cln(self.n2, x))))
        k = F.relu(self.skip(obs))
        h = torch.cat([x, k], dim=1)
        s = torch.sigmoid(self.sf(h))
        c = F.relu(self.ctx(x))
        a = self.anchor(h)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mn, mx = mm_pool(s, dy, dx)
            cp = mm_pool(c, dy, dx)
            ca = av_pool(c, dy, dx)
            z = F.relu(self.mix[i](torch.cat([mn, mx, cp[1], ca], dim=1)))
            res.append(self.outs[i](z) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        pooled = torch.cat([s.amax((2, 3)), s.mean((2, 3)), h.amax((2, 3)), h.mean((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.vnet(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
