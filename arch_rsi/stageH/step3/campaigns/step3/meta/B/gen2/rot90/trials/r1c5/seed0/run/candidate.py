import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
OBS_SYM = False
USE_LN = True
OUT_GAIN = 0.01
STEM = 64
WIDTH = 40
HEAD = 32
HID = 64


def relay(m, t, fy, fx):
    if t:
        m = m.transpose(2, 3)
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    return m


def unlay(m, t, fy, fx):
    if fx:
        m = m.flip(3)
    if fy:
        m = m.flip(2)
    if t:
        m = m.transpose(2, 3)
    return m


class CLN(nn.Module):
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
        P = self.Y * self.X
        sq = self.Y == self.X
        lays = [l for l in LAYOUTS if sq or l[0] == 0]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.layouts = lays
        self.exact = (self.n_place % P == 0)
        self.K = self.n_place // P if self.exact else -(-self.n_place // P)
        if OBS_SYM:
            self.syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or t == 0]
        else:
            self.syms = [(0, 0, 0)]
        c_obs = int(spec.in_channels) * len(self.syms) + 2
        if self.exact:
            c_mask = self.K * len(self.layouts)
            self.mask_embed = None
            self.out_lin = None
        else:
            c_mask = self.K
            self.mask_embed = nn.Linear(self.n_place, self.K * P)
            self.out_lin = nn.Linear(self.K * P, self.n_place)
        if USE_LN:
            self.obs_stem = nn.Conv2d(c_obs, STEM, 1)
            self.obs_ln = CLN(STEM)
            self.mask_stem = nn.Conv2d(c_mask, STEM, 1)
            self.mask_ln = CLN(STEM)
            self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
            self.stem_ln = CLN(WIDTH)
            self.ln1 = CLN(WIDTH)
            self.ln2 = CLN(WIDTH)
        else:
            self.stem1 = nn.Conv2d(c_obs + c_mask, STEM, 1)
            self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.out = nn.Conv2d(HEAD, self.K * len(self.layouts), 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, HID), nn.ReLU())
        self.done_head = nn.Linear(HID, 1)
        self.value_head = nn.Linear(HID, 1)
        if OUT_GAIN > 0:
            nn.init.orthogonal_(self.out.weight, gain=OUT_GAIN)
            nn.init.zeros_(self.out.bias)
            nn.init.orthogonal_(self.done_head.weight, gain=OUT_GAIN)
            nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        views = [relay(obs, s[0], s[1], s[2]) for s in self.syms]
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).reshape(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).reshape(1, 1, 1, X).expand(B, 1, Y, X)
        o = torch.cat(views + [ys, xs], 1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.exact:
            mm = mp.reshape(B, K, Y, X)
            mi = torch.cat([unlay(mm, l[0], l[1], l[2]) for l in self.layouts], 1)
        else:
            mi = self.mask_embed(mp).reshape(B, K, Y, X)
        if USE_LN:
            a = F.relu(self.obs_ln(self.obs_stem(o)))
            b = F.relu(self.mask_ln(self.mask_stem(mi)))
            s = F.relu(self.stem_ln(self.stem2(a + b)))
            f = F.relu(self.ln1(s + self.r1(s)))
            f = F.relu(self.ln2(f + self.r2(f)))
        else:
            s = F.relu(self.stem1(torch.cat([o, mi], 1)))
            s = F.relu(self.stem2(s))
            f = F.relu(s + self.r1(s))
            f = F.relu(f + self.r2(f))
        h = F.relu(self.h3(f) + self.h1(f))
        maps = self.out(h)
        logit = None
        for i, l in enumerate(self.layouts):
            m = relay(maps[:, i * K:(i + 1) * K], l[0], l[1], l[2]).reshape(B, -1)
            logit = m if logit is None else logit + m
        if not self.exact:
            logit = self.out_lin(logit)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([logit, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
