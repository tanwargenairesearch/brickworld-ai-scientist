import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
OBS_SYM = True
USE_LN = True
OUT_GAIN = 0.01

STEM = 64
MSTEM = 32
WIDTH = 40
HEAD = 32
GLOB = 64


def fwd(t, lay):
    tr, fy, fx = lay
    if tr:
        t = t.transpose(-2, -1)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def inv(t, lay):
    tr, fy, fx = lay
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-2, -1)
    return t


class CLN(nn.Module):
    def __init__(self, c, on):
        super().__init__()
        self.on = on
        self.ln = nn.LayerNorm(c) if on else nn.Identity()

    def forward(self, x):
        if not self.on:
            return x
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        self.K = (self.n_place + cells - 1) // cells
        self.padn = self.K * cells - self.n_place
        sq = self.Y == self.X
        self.lays = [l for l in LAYOUTS if sq or l[0] == 0]
        if OBS_SYM:
            self.syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or t == 0]
        else:
            self.syms = [(0, 0, 0)]
        nl = len(self.lays)
        c_obs = spec.in_channels * len(self.syms) + 2
        c_mask = self.K * nl + 2
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), CLN(STEM, True), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(c_mask, MSTEM, 1), CLN(MSTEM, True), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(STEM + MSTEM, WIDTH, 1), CLN(WIDTH, True), nn.ReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(c_obs + self.K * nl, STEM, 1), nn.ReLU(),
                                      nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n1 = CLN(WIDTH, USE_LN)
        self.n2 = CLN(WIDTH, USE_LN)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, self.K * nl, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if OUT_GAIN > 0:
            nn.init.orthogonal_(self.hout.weight, OUT_GAIN)
            nn.init.zeros_(self.hout.bias)
            nn.init.orthogonal_(self.done_head.weight, OUT_GAIN)
            nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        coords = torch.cat([yy, xx], dim=1)
        views = [fwd(obs, s) for s in self.syms]
        m = mask[:, :self.n_place].to(obs.dtype)
        if self.padn > 0:
            m = F.pad(m, (0, self.padn))
        m = m.reshape(B, K, Y, X)
        mv = [inv(m, l) for l in self.lays]
        if USE_LN:
            a = self.obs_stem(torch.cat(views + [coords], dim=1))
            b = self.mask_stem(torch.cat(mv + [coords], dim=1))
            f = self.stem2(torch.cat([a, b], dim=1))
        else:
            f = self.stem(torch.cat(views + [coords] + mv, dim=1))
        f = F.relu(self.n1(f + self.r1(f)))
        f = F.relu(self.n2(f + self.r2(f)))
        h = F.relu(self.h3(f) + self.h1(f))
        o = self.hout(h)
        place = None
        for i, l in enumerate(self.lays):
            p = fwd(o[:, i * K:(i + 1) * K], l).reshape(B, -1)
            place = p if place is None else place + p
        place = place[:, :self.n_place]
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
