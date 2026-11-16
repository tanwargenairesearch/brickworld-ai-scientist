import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.0015, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 2, 'ent_coef': 0.05, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2}

STEM = 64
W = 32
NB = 2
SKIP = 24
PROJ = 32
HID = 32
APROJ = 16
MIX = 32
PT = 8
G = 64
CAP = 8.0
KR = 0
TOK = 0


def orth(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def coord_channels(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([ys, xs], dim=1)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class TokenMix(nn.Module):
    def __init__(self, c, n, hid):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.l1 = nn.Linear(n, hid)
        self.l2 = nn.Linear(hid, n)

    def forward(self, x):
        b, c, y, xx = x.shape
        h = self.norm(x).reshape(b, c, y * xx)
        h = self.l2(F.relu(self.l1(h)))
        return x + h.reshape(b, c, y, xx)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        n = int(spec.Y) * int(spec.X)
        self.n = n
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        t = len(self.fps)
        self.nt = t
        self.shapes = sorted(set([f for f in self.fps if f[0] * f[1] > 1]))
        self.stem = nn.Sequential(nn.Conv2d(cin + 2, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, W, 1))
        layers = []
        for i in range(NB):
            layers.append(Block(W))
            if TOK > 0 and i < NB - 1:
                layers.append(TokenMix(W, n, TOK))
        self.trunk = nn.Sequential(*layers)
        self.nf = nn.GroupNorm(4, W)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        fc = W + SKIP
        self.proj = nn.Conv2d(fc, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.aproj = nn.Conv2d(fc, APROJ, 1)
        self.amix = nn.Conv2d(APROJ * (1 + 2 * len(self.shapes)), MIX, 1)
        self.tconv = nn.Conv2d(MIX, t * PT, 1)
        self.g5 = nn.Conv2d(t * PT, t * PT, 5, padding=2, groups=t)
        self.gout = nn.Conv2d(t * PT, t, 1, groups=t)
        self.anchor = nn.Conv2d(fc, 1, 1)
        if KR > 0:
            self.rfeat = nn.Conv2d(t * PT, t * KR, 1, groups=t)
            self.remap = nn.Linear(t * KR * n, spec.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * fc, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.type_bias = nn.Linear(G, t)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.modules():
            if isinstance(m, TokenMix):
                orth(m.l2, 1.0)
        for m in self.outs:
            orth(m, 0.01)
        orth(self.gout, 0.01)
        orth(self.anchor, 0.01)
        if KR > 0:
            nn.init.zeros_(self.remap.weight)
            nn.init.zeros_(self.remap.bias)
        orth(self.type_bias, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        x = self.trunk(self.stem(torch.cat([obs, coord_channels(obs)], dim=1)))
        f = torch.cat([F.relu(self.nf(x)), F.relu(self.skip(obs))], dim=1)
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = fp_pool(p, dy, dx)
            h = F.relu(self.mix[i](torch.cat([mx, mn, av], dim=1)))
            res.append(self.outs[i](h))
        own = torch.cat(res, dim=1)
        a = self.aproj(f)
        feats = [a]
        for dy, dx in self.shapes:
            mx, mn, av = fp_pool(a, dy, dx)
            feats.append(mx)
            feats.append(mn)
        u = F.relu(self.amix(torch.cat(feats, dim=1)))
        v = F.relu(self.tconv(u))
        v = F.relu(v + self.g5(v))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        tb = self.type_bias(g).view(B, self.nt, 1, 1)
        sc = own + self.gout(v) + self.anchor(f) + tb
        place = sc.flatten(1)
        if KR > 0:
            r = F.relu(self.rfeat(v)).flatten(1)
            place = place + self.remap(r)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if CAP > 0:
            logits = CAP * torch.tanh(logits / CAP)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
