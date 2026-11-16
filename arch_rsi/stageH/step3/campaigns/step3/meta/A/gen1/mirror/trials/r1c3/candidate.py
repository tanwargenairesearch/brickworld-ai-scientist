import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 256, 'rollout_len': 16, 'n_minibatches': 8, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95}

CM = 96
K = 4
LH = 32
COL = 48
WIDTH = 40
PROJ = 24
HID = 24
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


def pool3(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class PoolHead(nn.Module):
    def __init__(self, c, fps, proj, hid):
        super().__init__()
        self.fps = fps
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = pool3(p, dy, dx)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X, Z = int(spec.Y), int(spec.X), int(spec.Z)
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.register_buffer('coord', make_coords(Y, X))
        self.register_buffer('levels', (torch.arange(Z, dtype=torch.float32) / Z).view(1, Z, 1, 1))
        cin = int(spec.in_channels) + 2
        self.col1 = nn.Conv2d(cin, CM, 1)
        self.col2 = nn.Conv2d(CM, CM, 1)
        self.gh = nn.Conv2d(CM, Z, 1)
        self.ga = nn.Conv2d(CM, K * Z, 1)
        self.lmix = nn.ModuleList([nn.Conv2d(3 * K + 1, LH, 1) for _ in self.fps])
        self.lout = nn.ModuleList([nn.Conv2d(LH, 1, 1) for _ in self.fps])
        self.down = nn.Conv2d(CM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.head = PoolHead(WIDTH, self.fps, PROJ, HID)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + CM, GW), nn.ReLU(), nn.Linear(GW, GW), nn.ReLU())
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        init_all(self)
        for o in self.lout:
            init_layer(o, 0.01)
        for o in self.head.outs:
            init_layer(o, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = obs.shape[2], obs.shape[3]
        Z = self.Z
        xin = torch.cat([obs, self.coord.expand(B, -1, -1, -1)], dim=1)
        c = F.relu(self.col2(F.relu(self.col1(xin))))
        g = torch.sigmoid(self.gh(c))
        a = torch.sigmoid(self.ga(c))
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                G = g
            else:
                G = F.max_pool2d(F.pad(g, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)
            amx, amn, aav = pool3(a, dy, dx)
            Gn = torch.cat([G[:, 1:], torch.zeros_like(G[:, :1])], dim=1)
            land = F.relu(G - Gn)
            lw = land.unsqueeze(1)
            feats = [(A.reshape(B, K, Z, Y, X) * lw).sum(2) for A in (amx, amn, aav)]
            lvl = (land * self.levels).sum(1, keepdim=True)
            h = F.relu(self.lmix[t](torch.cat(feats + [lvl], dim=1)))
            outs.append(self.lout[t](h))
        lat = torch.cat(outs, dim=1)
        s = F.relu(self.down(c))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        place = lat + self.head(f) + self.type_bias.view(1, -1, 1, 1)
        g2 = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), c.amax((2, 3))], dim=1))
        logits = torch.cat([place.flatten(1), self.done_head(g2)], dim=1)
        return logits, self.value_head(g2).squeeze(1)


def build_policy(spec):
    return Net(spec)
