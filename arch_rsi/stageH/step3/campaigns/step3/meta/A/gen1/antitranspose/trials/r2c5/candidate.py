import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 1e-3, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 64, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2}

COLH = 128
COL = 64
W = 32
PROJ = 32
HID = 24
DCH = 8
DHID = 256
G = 4
DONE_PRIOR = -3.0


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Res(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.n1 = nn.GroupNorm(G, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.n2 = nn.GroupNorm(G, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class FootHead(nn.Module):
    def __init__(self, c, gdim, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        n = len(self.fps)
        self.proj = nn.Conv2d(c, proj, 1)
        self.gproj = nn.Linear(gdim, proj)
        self.foot = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1, bias=False) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        self.inter = nn.Conv2d(c, n, 1)
        self.anchor = nn.Conv2d(c, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(n))

    def forward(self, f, g):
        p = self.proj(f) + self.gproj(g)[:, :, None, None]
        res = []
        for (dy, dx), foot, mix, out in zip(self.fps, self.foot, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(foot(q) + mix(torch.cat([mx, mn], dim=1)))
            res.append(out(h))
        s = torch.cat(res, dim=1) + self.inter(f) + self.anchor(f)
        s = s + self.tbias.view(1, -1, 1, 1)
        return s.flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.col = nn.Sequential(
            nn.Conv2d(cin, COLH, 1), nn.GroupNorm(G, COLH), nn.ReLU(),
            nn.Conv2d(COLH, COL, 1), nn.ReLU())
        self.inp = nn.Conv2d(COL, W, 1)
        self.r1 = Res(W, 1)
        self.r2 = Res(W, 2)
        self.norm = nn.GroupNorm(G, W)
        c = COL + W
        gdim = 2 * c
        self.glob = nn.Sequential(nn.Linear(gdim, 64), nn.ReLU())
        self.head = FootHead(c, 64, spec.footprints, PROJ, HID)
        self.dproj = nn.Conv2d(c, DCH, 1)
        self.dense = nn.Sequential(nn.Linear(DCH * spec.Y * spec.X, DHID), nn.ReLU(), nn.Linear(DHID, spec.n_place))
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(gdim, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        for m in self.head.outs:
            init_layer(m, 0.01)
        init_layer(self.head.inter, 0.01)
        init_layer(self.head.anchor, 0.01)
        init_layer(self.dense[2], 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c0 = self.col(obs)
        x = self.r2(self.r1(self.inp(c0)))
        f = F.relu(self.norm(x))
        h = torch.cat([c0, f], dim=1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        place = self.head(h, g) + self.dense(F.relu(self.dproj(h)).flatten(1))
        d = self.done_head(g) + DONE_PRIOR
        lp = F.logsigmoid(-d) + F.log_softmax(place, dim=1)
        logits = torch.cat([lp, F.logsigmoid(d)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
