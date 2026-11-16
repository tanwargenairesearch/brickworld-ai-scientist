import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95}

STEM = 64
W = 40
CW = 64
SKIP = 16
P = 16
HH = 32
GOBS = 16
GW = 64


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def init_all(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_layer(m, math.sqrt(2.0))


def make_coords(Y, X):
    yy = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
    xx = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
    return torch.cat([yy, xx], dim=1).contiguous()


class ConcatHead(nn.Module):
    def __init__(self, c, fps, p, hh):
        super().__init__()
        self.fps = fps
        self.proj = nn.Conv2d(c, p, 1)
        self.m1 = nn.ModuleList([nn.Conv2d(p * dy * dx, hh, 1) for dy, dx in fps])
        self.m2 = nn.ModuleList([nn.Conv2d(hh, hh, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(hh, 1, 1) for _ in fps])

    def forward(self, f):
        p = self.proj(f)
        Y, X = p.shape[2], p.shape[3]
        res = []
        for (dy, dx), m1, m2, out in zip(self.fps, self.m1, self.m2, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts = [q[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
            h = F.relu(m1(torch.cat(parts, dim=1)))
            h = F.relu(m2(h))
            res.append(out(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.register_buffer('coord', make_coords(Y, X))
        cin = int(spec.in_channels) + 2
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.ns = nn.GroupNorm(4, STEM)
        self.stem2 = nn.Conv2d(STEM, W, 1)
        self.b1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(4, W)
        self.down = nn.Conv2d(W, CW, 3, stride=2, padding=1)
        self.nd = nn.GroupNorm(4, CW)
        self.cc1 = nn.Conv2d(CW, CW, 3, padding=1)
        self.cn1 = nn.GroupNorm(4, CW)
        self.cc2 = nn.Conv2d(CW, CW, 3, padding=1)
        self.cn2 = nn.GroupNorm(4, CW)
        self.up = nn.Conv2d(CW, W, 1)
        self.merge = nn.Conv2d(2 * W, W, 1)
        self.nm = nn.GroupNorm(4, W)
        self.b2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(4, W)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        hc = W + SKIP
        self.head = ConcatHead(hc, self.fps, P, HH)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.gobs = nn.Conv2d(cin, GOBS, 1)
        self.glob = nn.Sequential(nn.Linear(2 * hc + 2 * CW + 2 * GOBS, GW), nn.ReLU(), nn.Linear(GW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        init_all(self)
        for o in self.head.outs:
            init_layer(o, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = obs.shape[2], obs.shape[3]
        xin = torch.cat([obs, self.coord.expand(B, -1, -1, -1)], dim=1)
        x = F.relu(self.stem2(F.relu(self.ns(self.stem1(xin)))))
        x = F.relu(x + self.n1(self.b1(x)))
        c = F.relu(self.nd(self.down(x)))
        c = F.relu(c + self.cn1(self.cc1(c)))
        c = F.relu(c + self.cn2(self.cc2(c)))
        u = F.relu(self.up(F.interpolate(c, size=(Y, X), mode='nearest')))
        x = F.relu(self.nm(self.merge(torch.cat([x, u], dim=1))))
        x = F.relu(x + self.n2(self.b2(x)))
        f = torch.cat([x, F.relu(self.skip(xin))], dim=1)
        place = self.head(f) + self.type_bias.view(1, -1, 1, 1)
        go = F.relu(self.gobs(xin))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), c.mean((2, 3)), c.amax((2, 3)), go.mean((2, 3)), go.amax((2, 3))], dim=1))
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
