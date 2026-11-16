import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
OBS_SYM = True
USE_LN = True
GCTX = True
DILATED = True
TYPE_MINOR = True
SEP_CRITIC = False
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
CRIT = 32

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
        c_mask = self.K * len(lays)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(c_mask, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), ChanLN(WIDTH), nn.ReLU())
        else:
            self.stem1 = nn.Sequential(nn.Conv2d(c_obs + c_mask, STEM, 1), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        dils = (1, 2, 1) if DILATED else (1, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in dils])
        if USE_LN:
            self.lns = nn.ModuleList([ChanLN(WIDTH) for _ in dils])
        if GCTX:
            self.gctx = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in dils])
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, self.K * len(lays), 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        if SEP_CRITIC:
            c_raw = spec.in_channels + 2
            if USE_LN:
                self.crit = nn.Sequential(nn.Conv2d(c_raw, CRIT, 1), ChanLN(CRIT), nn.ReLU(), nn.Conv2d(CRIT, CRIT, 3, padding=1), nn.ReLU())
            else:
                self.crit = nn.Sequential(nn.Conv2d(c_raw, CRIT, 1), nn.ReLU(), nn.Conv2d(CRIT, CRIT, 3, padding=1), nn.ReLU())
            self.crit_fc = nn.Sequential(nn.Linear(2 * CRIT, GH), nn.ReLU(), nn.Linear(GH, 1))
        if USE_LN:
            for m in (self.hout, self.done_head):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.value_head.weight, 1.0)
            nn.init.zeros_(self.value_head.bias)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], dim=1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        cells = self.Y * self.X
        m = mask[:, :self.n_place].float()
        pad = self.K * cells - self.n_place
        if pad > 0:
            m = F.pad(m, (0, pad))
        if TYPE_MINOR:
            m = m.view(B, self.Y, self.X, self.K).permute(0, 3, 1, 2)
        else:
            m = m.view(B, self.K, self.Y, self.X)
        mfeat = torch.cat([fwd_layout(m, l) for l in self.lays], dim=1)
        crd = self.coords.expand(B, -1, -1, -1)
        ofeat = torch.cat([fwd_layout(obs, s) for s in self.syms] + [crd], dim=1)
        if USE_LN:
            s = self.obs_stem(ofeat) + self.mask_stem(mfeat)
        else:
            s = self.stem1(torch.cat([ofeat, mfeat], dim=1))
        x = self.stem2(s)
        for i, conv in enumerate(self.res):
            y = x + conv(x)
            if GCTX:
                g0 = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
                y = y + self.gctx[i](g0)[:, :, None, None]
            if USE_LN:
                y = self.lns[i](y)
            x = F.relu(y)
        h = F.relu(self.h3(x) + self.h1(x))
        o = self.hout(h).view(B, len(self.lays), self.K, self.Y, self.X)
        place = 0
        for i, l in enumerate(self.lays):
            place = place + inv_layout(o[:, i], l)
        if TYPE_MINOR:
            place = place.permute(0, 2, 3, 1)
        place = place.reshape(B, -1)[:, :self.n_place]
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if SEP_CRITIC:
            c = self.crit(torch.cat([obs, crd], dim=1))
            v = self.crit_fc(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1)).squeeze(1)
        else:
            v = self.value_head(g).squeeze(1)
        return logits, v


def build_policy(spec):
    return Net(spec)
