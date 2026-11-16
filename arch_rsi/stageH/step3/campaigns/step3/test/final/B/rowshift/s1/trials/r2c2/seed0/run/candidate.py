import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [(0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
TYPE_MINOR = True
OBS_SYM = False
USE_LN = False
SMALL_OUT = False
GLOB_CTX = False
DILATED = False
SEP_CRITIC = False
STEM = 64
WIDTH = 40
HEADC = 32
CRIT = 32
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


def fwd_layout(t, tr, fy, fx):
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def inv_layout(t, tr, fy, fx):
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.n_place = int(spec.n_place)
        yx = self.Y * self.X
        self.K = (self.n_place + yx - 1) // yx
        sq = self.Y == self.X
        self.layouts = [l for l in LAYOUTS if sq or not l[0]]
        if OBS_SYM:
            self.syms = [(a, b, c) for a in ((0, 1) if sq else (0,)) for b in (0, 1) for c in (0, 1)]
        else:
            self.syms = [(0, 0, 0)]
        L = len(self.layouts)
        cin_obs = spec.in_channels * len(self.syms) + 2
        cin_mask = self.K * L
        if USE_LN:
            self.obs_stem = nn.Conv2d(cin_obs, STEM, 1)
            self.obs_ln = CLN(STEM)
            self.mask_stem = nn.Conv2d(cin_mask, STEM, 1)
            self.mask_ln = CLN(STEM)
            self.stem2 = nn.Conv2d(2 * STEM, WIDTH, 1)
        else:
            self.stem1 = nn.Conv2d(cin_obs + cin_mask, STEM, 1)
            self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        dils = [1, 1] + ([2] if DILATED else [])
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in dils])
        self.res_ln = nn.ModuleList([CLN(WIDTH) if USE_LN else nn.Identity() for _ in dils])
        if GLOB_CTX:
            self.ctx = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in dils])
        self.h3 = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADC, 1)
        self.hout = nn.Conv2d(HEADC, self.K * L, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        if SEP_CRITIC:
            self.c_stem = nn.Conv2d(spec.in_channels + 2, CRIT, 1)
            self.c_ln = CLN(CRIT)
            self.c_conv = nn.Conv2d(CRIT, CRIT, 3, padding=1)
            self.c_hid = nn.Linear(2 * CRIT, 64)
            self.value_head = nn.Linear(64, 1)
        else:
            self.value_head = nn.Linear(64, 1)
        if SMALL_OUT:
            for m in (self.hout, self.done_head, self.value_head):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        yx = self.Y * self.X
        K = self.K
        mp = mask[:, :self.n_place].float()
        pad = K * yx - self.n_place
        if pad > 0:
            mp = F.pad(mp, (0, pad))
        if TYPE_MINOR:
            mp = mp.reshape(B, self.Y, self.X, K).permute(0, 3, 1, 2)
        else:
            mp = mp.reshape(B, K, self.Y, self.X)
        coords = self.coords.expand(B, -1, -1, -1)
        mfeat = torch.cat([fwd_layout(mp, *l) for l in self.layouts], 1)
        ofeat = torch.cat([fwd_layout(obs, *s) for s in self.syms] + [coords], 1)
        if USE_LN:
            o = F.relu(self.obs_ln(self.obs_stem(ofeat)))
            m = F.relu(self.mask_ln(self.mask_stem(mfeat)))
            f = F.relu(self.stem2(torch.cat([o, m], 1)))
        else:
            f = F.relu(self.stem1(torch.cat([ofeat, mfeat], 1)))
            f = F.relu(self.stem2(f))
        for i in range(len(self.res)):
            u = self.res[i](self.res_ln[i](f))
            if GLOB_CTX:
                c = self.ctx[i](torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
                u = u + c[:, :, None, None]
            f = F.relu(f + u)
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        place = 0
        for i, l in enumerate(self.layouts):
            mo = inv_layout(out[:, i * K:(i + 1) * K], *l)
            if TYPE_MINOR:
                mo = mo.permute(0, 2, 3, 1)
            place = place + mo.reshape(B, -1)
        place = place[:, :self.n_place]
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        if SEP_CRITIC:
            c = F.relu(self.c_ln(self.c_stem(torch.cat([obs, coords], 1))))
            c = F.relu(c + self.c_conv(c))
            v = self.value_head(F.relu(self.c_hid(torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1))))
        else:
            v = self.value_head(g)
        return logits, v.squeeze(1)


def build_policy(spec):
    return Net(spec)
