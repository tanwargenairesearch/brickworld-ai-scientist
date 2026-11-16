import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.003, 'n_envs': 1024, 'rollout_len': 16, 'n_minibatches': 32, 'ppo_epochs': 1, 'ent_coef': 0.02, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.3}

D = 32
NMIX = 2
TOK = 32
TOKG = 1
CHMULT = 2
CHREMAP = True
GLU = False
CONV = 0
GROUPS = 8
PROJ = 16
SKIP = 16
FOOT = True
HID = 16
DENSE3 = True
FLATK = 2
GLOB = 64


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def unif(shape, fan_in):
    b = 1.0 / math.sqrt(fan_in)
    return nn.Parameter(torch.empty(*shape).uniform_(-b, b))


class TokenMix(nn.Module):
    def __init__(self, d, n, t, g):
        super().__init__()
        self.g = g
        self.w1 = unif((g, n, t), n)
        self.b1 = nn.Parameter(torch.zeros(g, 1, t))
        self.w2 = unif((g, t, n), t)
        self.b2 = nn.Parameter(torch.zeros(g, 1, n))

    def forward(self, y):
        B, d, n = y.shape
        y = y.reshape(B, self.g, d // self.g, n)
        h = F.gelu(torch.einsum('bgdn,gnt->bgdt', y, self.w1) + self.b1)
        o = torch.einsum('bgdt,gtn->bgdn', h, self.w2) + self.b2
        return o.reshape(B, d, n)


class MixBlock(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.tok = TokenMix(d, n, TOK, TOKG)
        self.ln2 = nn.LayerNorm(d)
        self.ch1 = nn.Linear(d, CHMULT * d)
        self.ch2 = nn.Linear(CHMULT * d, d)

    def forward(self, x):
        y = self.ln1(x).transpose(1, 2)
        x = x + self.tok(y).transpose(1, 2)
        x = x + self.ch2(F.gelu(self.ch1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        N = self.Y * self.X
        self.N = N
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.nt = nt
        cin = int(spec.in_channels)
        g2 = math.sqrt(2.0)
        sin = cin
        if CHREMAP:
            self.remap_w = unif((cin, N, N), N)
            self.remap_b = nn.Parameter(torch.zeros(cin, N))
            sin = 2 * cin
        self.s1 = nn.Linear(sin, 2 * D if GLU else D)
        self.pos = nn.Parameter(torch.randn(N, D) * 0.02)
        self.s2 = nn.Linear(D, D)
        self.blocks = nn.ModuleList([MixBlock(D, N) for _ in range(NMIX)])
        self.lnf = nn.LayerNorm(D)
        self.convs = nn.ModuleList([ortho(nn.Conv2d(D, D, 3, padding=1), g2) for _ in range(CONV)])
        self.norms = nn.ModuleList([nn.GroupNorm(GROUPS, D) for _ in range(CONV)])
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), g2)
        self.proj = ortho(nn.Conv2d(D, PROJ, 1), g2)
        P = PROJ + SKIP
        if FOOT:
            self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * P, HID, 1), g2) for _ in self.fps])
            self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
            self.anchor = ortho(nn.Conv2d(P, 1, 1), 0.01)
            self.ctx = ortho(nn.Linear(GLOB, nt * HID), 1.0)
        if DENSE3:
            self.dense3 = ortho(nn.Conv2d(P, nt, 3, padding=1), 0.01)
        if FLATK > 0:
            self.flat_in = ortho(nn.Conv2d(P, nt * FLATK, 1), 1.0)
            self.flat_w = nn.Parameter(torch.randn(nt, FLATK, N, N) * (0.01 / math.sqrt(FLATK * N)))
        self.type_bias = nn.Parameter(torch.zeros(1, nt, 1, 1))
        self.glob = nn.Sequential(ortho(nn.Linear(2 * D + 2 * P, GLOB), g2), nn.ReLU())
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        o = obs.flatten(2)
        if CHREMAP:
            r = torch.einsum('bcn,cnm->bcm', o, self.remap_w) + self.remap_b
            z = torch.cat([o, r], 1)
        else:
            z = o
        h = self.s1(z.transpose(1, 2))
        if GLU:
            a, b = h.chunk(2, -1)
            h = a * torch.sigmoid(b)
        else:
            h = F.relu(h)
        h = F.relu(self.s2(h + self.pos))
        for blk in self.blocks:
            h = blk(h)
        h = self.lnf(h)
        f = h.transpose(1, 2).reshape(B, D, self.Y, self.X)
        for conv, nrm in zip(self.convs, self.norms):
            f = F.relu(f + nrm(conv(f)))
        s = F.relu(self.skip(obs))
        p = torch.cat([F.relu(self.proj(f)), s], 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), p.mean((2, 3)), p.amax((2, 3))], 1))
        place = self.type_bias.expand(B, self.nt, self.Y, self.X)
        if FOOT:
            ctx = self.ctx(g).view(B, self.nt, HID, 1, 1)
            res = []
            for i, (dy, dx) in enumerate(self.fps):
                if dy * dx == 1:
                    mx, mn, av = p, p, p
                else:
                    q = F.pad(p, (0, dx - 1, 0, dy - 1))
                    mx = F.max_pool2d(q, (dy, dx), stride=1)
                    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                    av = F.avg_pool2d(q, (dy, dx), stride=1)
                hh = F.relu(self.mix[i](torch.cat([mx, mn, av], 1)) + ctx[:, i])
                res.append(self.outs[i](hh))
            place = place + torch.cat(res, 1) + self.anchor(p)
        if DENSE3:
            place = place + self.dense3(p)
        place = place.reshape(B, -1)
        if FLATK > 0:
            q = self.flat_in(p).reshape(B, self.nt, FLATK, self.N)
            place = place + torch.einsum('btkn,tknm->btm', q, self.flat_w).reshape(B, -1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
