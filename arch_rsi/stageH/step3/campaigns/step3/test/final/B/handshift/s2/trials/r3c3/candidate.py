import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = ()
SPATIAL = False
OBS_SYM = True
USE_LN = True
MASKC = 8
STEM = 64
WIDTH = 40
FC = 16
HEAD = 32
GH = 64

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


def fwd_layout(t, lay):
    tr, fy, fx = lay
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def inv_layout(t, lay):
    tr, fy, fx = lay
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        cells = self.Y * self.X
        self.K = (self.n_place + cells - 1) // cells
        sq = self.Y == self.X
        lays = []
        for l in LAYOUTS:
            l2 = tuple(l) if sq else (0, l[1], l[2])
            if l2 not in lays:
                lays.append(l2)
        self.lays = lays
        if OBS_SYM:
            syms = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if sq or a == 0]
        else:
            syms = [(0, 0, 0)]
        self.syms = syms
        c_obs = spec.in_channels * len(syms) + 2
        self.mask_fc = nn.Linear(self.n_actions, MASKC * cells)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(MASKC, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), ChanLN(WIDTH), nn.ReLU())
            self.lns = nn.ModuleList([ChanLN(WIDTH) for _ in range(2)])
        else:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(MASKC, STEM, 1), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(2)])
        self.flat = nn.Conv2d(WIDTH, FC, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.pi = nn.Linear(FC * cells + GH, self.n_actions)
        self.value_head = nn.Linear(GH, 1)
        if SPATIAL:
            self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
            self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
            self.hout = nn.Conv2d(HEAD, self.K * len(lays), 1)
            nn.init.orthogonal_(self.hout.weight, 0.01)
            nn.init.zeros_(self.hout.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], dim=1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        mm = self.mask_fc(mask.float()).view(B, MASKC, self.Y, self.X)
        crd = self.coords.expand(B, -1, -1, -1)
        ofeat = torch.cat([fwd_layout(obs, s) for s in self.syms] + [crd], dim=1)
        x = self.stem2(self.obs_stem(ofeat) + self.mask_stem(mm))
        for i, conv in enumerate(self.res):
            y = x + conv(x)
            if USE_LN:
                y = self.lns[i](y)
            x = F.relu(y)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        f = F.relu(self.flat(x)).reshape(B, -1)
        logits = self.pi(torch.cat([f, g], dim=1))
        if SPATIAL:
            h = F.relu(self.h3(x) + self.h1(x))
            o = self.hout(h).view(B, len(self.lays), self.K, self.Y, self.X)
            place = 0
            for i, l in enumerate(self.lays):
                place = place + inv_layout(o[:, i], l)
            place = place.reshape(B, -1)[:, :self.n_place]
            logits = torch.cat([logits[:, :self.n_place] + place, logits[:, self.n_place:]], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
