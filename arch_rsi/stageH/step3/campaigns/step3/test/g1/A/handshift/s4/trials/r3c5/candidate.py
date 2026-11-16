import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.003, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 1, 'ent_coef': 0.02, 'gamma': 0.995, 'gae_lambda': 0.99, 'clip_eps': 0.3}

W = 48
COORD = True
INLN = True
USE_C0 = True
NB = 0
P = 24
K = 16
HID = 16
LOCAL3 = 16
FLATD = 8
GLOB = 64
GROUPS = 8


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class CellLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def fpool(q, dy, dx):
    if dy * dx == 1:
        return q, q, q
    r = F.pad(q, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(r, (dy, dx), stride=1)
    mn = -F.max_pool2d(-r, (dy, dx), stride=1)
    av = F.avg_pool2d(r, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.nt = nt
        cin = int(spec.in_channels) + (2 if COORD else 0)
        g = math.sqrt(2.0)
        self.s1 = ortho(nn.Conv2d(cin, W, 1), g)
        self.ln1 = CellLN(W) if INLN else nn.Identity()
        self.s2 = ortho(nn.Conv2d(W, W, 1), g)
        if USE_C0:
            self.c0 = ortho(nn.Conv2d(W, W, 3, padding=1), g)
            self.n0 = nn.GroupNorm(GROUPS, W)
        self.blocks = nn.ModuleList([ortho(nn.Conv2d(W, W, 3, padding=1), g) for _ in range(NB)])
        self.norms = nn.ModuleList([nn.GroupNorm(GROUPS, W) for _ in range(NB)])
        self.proj = ortho(nn.Conv2d(W, P, 1), g)
        self.skip = ortho(nn.Conv2d(cin, P, 1), g)
        PP = 2 * P
        hin = 3 * PP
        gin = 2 * W + 2 * PP
        self.KK = 2 * K
        if K > 0:
            self.l1 = ortho(nn.Conv2d(8, K, 1), g)
            self.l2 = ortho(nn.Conv2d(K, K, 1), g)
            self.land = nn.ModuleList([ortho(nn.Linear(3 * self.KK, 1), 1.0) for _ in self.fps])
            hin += 3 * self.KK
            gin += 2 * K
        self.mix = nn.ModuleList([ortho(nn.Conv2d(hin, HID, 1), g) for _ in self.fps])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.type_bias = nn.Parameter(torch.zeros(1, nt, 1, 1))
        if LOCAL3 > 0:
            self.loc1 = ortho(nn.Conv2d(PP, LOCAL3, 3, padding=1), g)
            self.loc2 = ortho(nn.Conv2d(LOCAL3, nt, 1), 0.01)
        if FLATD > 0:
            N = self.Y * self.X
            self.flat_in = ortho(nn.Conv2d(PP, FLATD, 1), g)
            self.flat_out = ortho(nn.Linear(FLATD * N, nt * N), 0.01)
        self.glob = nn.Sequential(ortho(nn.Linear(gin, GLOB), g), nn.ReLU())
        self.ctx = ortho(nn.Linear(GLOB, nt * HID), 1.0)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, Z = self.Y, self.X, self.Z
        KK = self.KK
        if COORD:
            yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
            xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
            x0 = torch.cat([obs, yy, xx], 1)
        else:
            x0 = obs
        h = F.relu(self.ln1(self.s1(x0)))
        h = F.relu(self.s2(h))
        f = h
        if USE_C0:
            f = F.relu(self.n0(self.c0(f)))
        for conv, nrm in zip(self.blocks, self.norms):
            f = F.relu(f + nrm(conv(f)))
        p = torch.cat([F.relu(self.proj(f)), F.relu(self.skip(x0))], 1)
        gparts = [f.mean((2, 3)), f.amax((2, 3)), p.mean((2, 3)), p.amax((2, 3))]
        v = None
        if K > 0:
            lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
            below = torch.cat([torch.zeros_like(obs[:, :1]), obs[:, :Z - 1]], 1).unsqueeze(1)
            hts = obs[:, 4 * Z:4 * Z + 2].unsqueeze(2).expand(B, 2, Z, Y, X)
            zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
            per = torch.cat([lay, below, hts, zc], 1)
            per = per.permute(0, 2, 1, 3, 4).reshape(B * Z, 8, Y, X)
            u = F.relu(self.l2(F.relu(self.l1(per)))).reshape(B, Z, K, Y, X)
            cm = torch.cummax(u, dim=1).values
            cb = torch.cat([torch.zeros_like(cm[:, :1]), cm[:, :-1]], 1)
            feats = torch.cat([u, cb], 2)
            v = feats.reshape(B, Z * KK, Y, X)
            gparts += [u.mean((1, 3, 4)), u.amax((1, 3, 4))]
        g = self.glob(torch.cat(gparts, 1))
        ctx = self.ctx(g).view(B, self.nt, HID, 1, 1)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = fpool(p, dy, dx)
            parts = [mx, mn, av]
            if K > 0:
                a1, a2, a3 = fpool(v, dy, dx)
                up = torch.cat([a1.reshape(B, Z, KK, Y, X), a2.reshape(B, Z, KK, Y, X), a3.reshape(B, Z, KK, Y, X)], 2)
                sc = self.land[i](up.permute(0, 1, 3, 4, 2)).squeeze(-1)
                att = torch.softmax(sc, dim=1).unsqueeze(2)
                parts.append((att * up).sum(1))
            hh = F.relu(self.mix[i](torch.cat(parts, 1)) + ctx[:, i])
            res.append(self.outs[i](hh))
        place = torch.cat(res, 1) + self.type_bias
        if LOCAL3 > 0:
            place = place + self.loc2(F.relu(self.loc1(p)))
        place = place.flatten(1)
        if FLATD > 0:
            place = place + self.flat_out(F.relu(self.flat_in(p)).flatten(1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
