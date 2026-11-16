import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 32, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2}

W = 64
BIL = 32
NRES = 0
TRUNK = 0
TW = 32
PROJ = 24
SKIP = 8
MARG = 0
HID = 48
HID2 = True
LOCAL3 = 0
FLATD = 0
MASKIN = False
GLOB = 64
GROUPS = 8


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class CellLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = CellLN(c)
        self.a = ortho(nn.Conv2d(c, 2 * c, 1), math.sqrt(2.0))
        self.b = ortho(nn.Conv2d(2 * c, c, 1), 1.0)

    def forward(self, x):
        return x + self.b(F.relu(self.a(self.ln(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.nt = nt
        self.nplace = int(spec.n_place)
        cin = int(spec.in_channels) + 2 + (nt if MASKIN else 0)
        g = math.sqrt(2.0)
        self.s1 = ortho(nn.Conv2d(cin, W, 1), g)
        self.ln1 = CellLN(W)
        self.s2 = ortho(nn.Conv2d(W, W, 1), g)
        self.res = nn.ModuleList([ResMLP(W) for _ in range(NRES)])
        fw = W
        if BIL > 0:
            self.ba = ortho(nn.Conv2d(W, BIL, 1), 1.0)
            self.bb = ortho(nn.Conv2d(W, BIL, 1), 1.0)
            fw = W + BIL
        if TRUNK > 0:
            self.t0 = ortho(nn.Conv2d(fw, TW, 3, padding=1), g)
            self.tn0 = nn.GroupNorm(GROUPS, TW)
            self.tb = nn.ModuleList([ortho(nn.Conv2d(TW, TW, 3, padding=1), g) for _ in range(TRUNK - 1)])
            self.tn = nn.ModuleList([nn.GroupNorm(GROUPS, TW) for _ in range(TRUNK - 1)])
            fw = fw + TW
        self.fw = fw
        self.proj = ortho(nn.Conv2d(fw, PROJ, 1), g)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), g)
        P = PROJ + SKIP
        self.heads = nn.ModuleList()
        self.h2 = nn.ModuleList()
        self.outs = nn.ModuleList()
        for (dy, dx) in self.fps:
            self.heads.append(ortho(nn.Conv2d(P, HID, (dy + 2 * MARG, dx + 2 * MARG)), g))
            if HID2:
                self.h2.append(ortho(nn.Conv2d(HID, HID, 1), g))
            self.outs.append(ortho(nn.Conv2d(HID, 1, 1), 0.01))
        self.anchor = ortho(nn.Conv2d(P, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(1, nt, 1, 1))
        if LOCAL3 > 0:
            self.loc1 = ortho(nn.Conv2d(P, LOCAL3, 3, padding=1), g)
            self.loc2 = ortho(nn.Conv2d(LOCAL3, nt, 1), 0.01)
        if FLATD > 0:
            N = self.Y * self.X
            self.flat_in = ortho(nn.Conv2d(P, FLATD, 1), g)
            self.flat_out = ortho(nn.Linear(FLATD * N, nt * N), 0.01)
        gin = 2 * fw + 2 * P
        self.glob = nn.Sequential(ortho(nn.Linear(gin, GLOB), g), nn.ReLU(), ortho(nn.Linear(GLOB, GLOB), g), nn.ReLU())
        self.ctx = ortho(nn.Linear(GLOB, nt * HID), 1.0)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts = [obs, yy, xx]
        if MASKIN:
            parts.append(mask[:, :self.nplace].to(obs.dtype).reshape(B, self.nt, Y, X))
        x0 = torch.cat(parts, 1)
        h = F.relu(self.ln1(self.s1(x0)))
        h = F.relu(self.s2(h))
        for blk in self.res:
            h = blk(h)
        f = h
        if BIL > 0:
            f = torch.cat([h, self.ba(h) * self.bb(h)], 1)
        if TRUNK > 0:
            t = F.relu(self.tn0(self.t0(f)))
            for conv, nrm in zip(self.tb, self.tn):
                t = F.relu(t + nrm(conv(t)))
            f = torch.cat([f, t], 1)
        p = torch.cat([F.relu(self.proj(f)), F.relu(self.skip(x0))], 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), p.mean((2, 3)), p.amax((2, 3))], 1))
        ctx = self.ctx(g).view(B, self.nt, HID, 1, 1)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(p, (MARG, dx - 1 + MARG, MARG, dy - 1 + MARG))
            hh = F.relu(self.heads[i](q) + ctx[:, i])
            if HID2:
                hh = F.relu(self.h2[i](hh))
            res.append(self.outs[i](hh))
        place = torch.cat(res, 1) + self.anchor(p) + self.type_bias
        if LOCAL3 > 0:
            place = place + self.loc2(F.relu(self.loc1(p)))
        place = place.flatten(1)
        if FLATD > 0:
            place = place + self.flat_out(F.relu(self.flat_in(p)).flatten(1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
