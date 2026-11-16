import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
SQ = 48
GH = 64
SCALE = 5.0


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        g = math.sqrt(2)
        self.s1 = init_layer(nn.Conv2d(cin, STEM, 1), g)
        self.s2 = init_layer(nn.Conv2d(STEM, COL, 1), g)
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1), g)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = init_layer(nn.Conv2d(cin, SKIP, 1), g)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1)


class Glob(nn.Module):
    def __init__(self, c, T):
        super().__init__()
        g = math.sqrt(2)
        self.l1 = init_layer(nn.Linear(2 * c, GH), g)
        self.l2 = init_layer(nn.Linear(GH, GH), g)
        self.done = init_layer(nn.Linear(GH, 1), 0.01)
        self.value = init_layer(nn.Linear(GH, 1), 1.0)
        self.tb = init_layer(nn.Linear(GH, T), 0.01)

    def forward(self, f):
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        g = F.relu(self.l2(F.relu(self.l1(g))))
        return g, self.done(g), self.value(g).squeeze(1), self.tb(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.K = max(max(a, b) for a, b in self.fps)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.sq = init_layer(nn.Conv2d(c, SQ, self.K), math.sqrt(2))
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(SQ, self.T, 1), 0.01) for _ in range(2)])
        self.tmix = nn.Parameter(torch.zeros(self.T, self.T))
        self.glob = Glob(c, self.T)
        self.dense = init_layer(nn.Linear(GH, int(spec.n_actions)), 0.01)
        self.alpha = nn.Parameter(torch.zeros(2))

    def forward(self, obs, mask):
        f = self.trunk(obs)
        K = self.K
        res = []
        for i, u in enumerate([f, f.transpose(2, 3)]):
            z = F.pad(u, (0, K - 1, 0, K - 1))
            h = F.relu(self.sq(z))
            res.append(self.outs[i](h).flatten(2))
        L = torch.stack(res, 1)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = (L * w.view(1, 2, 1, 1)).sum(1)
        place = place + torch.einsum('ts,bsn->btn', self.tmix, place)
        g, done, value, tb = self.glob(f)
        place = place + tb.unsqueeze(2)
        logits = torch.cat([place.flatten(1), done], 1) + self.dense(g)
        return logits, value


def build_policy(spec):
    return Net(spec)
