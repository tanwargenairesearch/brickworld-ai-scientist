import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 3, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.005}

H = 64
CT = 24
SK = 16
K = 24
HID = 24
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        T = len(self.fps)
        self.e1 = ortho(nn.Conv2d(C, H, 1))
        self.n1 = nn.LayerNorm(H)
        self.e2 = ortho(nn.Conv2d(H, H, 1), 1.0)
        self.n2 = nn.LayerNorm(3 * H)
        self.rc = ortho(nn.Conv2d(3 * H, H, 1), 1.0)
        self.ctx = ortho(nn.Conv2d(H, CT, 3, padding=1))
        self.skip = ortho(nn.Conv2d(C, SK, 1))
        D = H + CT + SK
        self.sf = ortho(nn.Conv2d(D, K, 1), 1.0)
        self.win = nn.ModuleList([ortho(nn.Conv2d(K, HID, (dy + 2, dx + 2))) for dy, dx in self.fps])
        self.mix = nn.ModuleList([ortho(nn.Conv2d(2 * K, HID, 1)) for t in range(T)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for t in range(T)])
        self.anchor = ortho(nn.Conv2d(D, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(T))
        P = 2 * K + 2 * D
        self.glob = nn.Sequential(ortho(nn.Linear(P, G)), nn.ReLU(), ortho(nn.Linear(G, G)), nn.ReLU())
        self.done = ortho(nn.Linear(G, 1), 0.01)
        self.vnet = nn.Sequential(ortho(nn.Linear(P, G)), nn.ReLU(), ortho(nn.Linear(G, 1), 1.0))

    def forward(self, obs, mask):
        x = F.relu(self.e1(obs))
        x = x + self.e2(F.relu(cln(self.n1, x)))
        rmax = x.amax(3, keepdim=True).expand_as(x)
        cmax = x.amax(2, keepdim=True).expand_as(x)
        x = F.relu(x + self.rc(F.relu(cln(self.n2, torch.cat([x, rmax, cmax], dim=1)))))
        c = F.relu(self.ctx(x))
        k = F.relu(self.skip(obs))
        h = torch.cat([x, c, k], dim=1)
        s = torch.sigmoid(self.sf(h))
        a = self.anchor(h)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(s, (1, dx, 1, dy))
            w = self.win[i](q)
            mn, mx = mm_pool(s, dy, dx)
            z = F.relu(w + self.mix[i](torch.cat([mn, mx], dim=1)))
            res.append(self.outs[i](z) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        pooled = torch.cat([s.amax((2, 3)), s.mean((2, 3)), h.amax((2, 3)), h.mean((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.vnet(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
