import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
OBS_SYM = True
USE_LN = False
STEM = 64
WIDTH = 40
HEADW = 32
MSTEM = 32


def fwd_lay(v, t):
    tr, fy, fx = t
    if tr:
        v = v.transpose(-1, -2)
    if fy:
        v = v.flip(-2)
    if fx:
        v = v.flip(-1)
    return v


def inv_lay(v, t):
    tr, fy, fx = t
    if fx:
        v = v.flip(-1)
    if fy:
        v = v.flip(-2)
    if tr:
        v = v.transpose(-1, -2)
    return v


class CellLN(nn.Module):
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
        hw = self.Y * self.X
        self.K = (self.n_place + hw - 1) // hw
        sq = self.Y == self.X
        self.lays = [l for l in LAYOUTS if sq or l[0] == 0]
        if len(self.lays) == 0:
            self.lays = [(0, 1, 1)]
        if OBS_SYM:
            self.syms = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1) if sq or a == 0]
        else:
            self.syms = [(0, 0, 0)]
        L = len(self.lays)
        cin_obs = spec.in_channels * len(self.syms) + 2
        cin_mask = self.K * L
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(cin_obs, STEM, 1), CellLN(STEM), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
            self.mask_stem = nn.Sequential(nn.Conv2d(cin_mask, MSTEM, 1), CellLN(MSTEM), nn.ReLU(), nn.Conv2d(MSTEM, WIDTH, 1))
            self.stem_ln = CellLN(WIDTH)
            self.n1 = CellLN(WIDTH)
            self.n2 = CellLN(WIDTH)
        else:
            self.stem = nn.Sequential(nn.Conv2d(cin_obs + cin_mask, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADW, 1)
        self.hout = nn.Conv2d(HEADW, self.K * L, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        if USE_LN:
            for m in (self.hout, self.done_head):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        hw = Y * X
        mp = mask[:, :self.n_place].to(obs.dtype)
        if K * hw > self.n_place:
            mp = F.pad(mp, (0, K * hw - self.n_place))
        mp = mp.reshape(B, K, Y, X)
        mviews = torch.cat([fwd_lay(mp, l) for l in self.lays], 1)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        oin = torch.cat([fwd_lay(obs, s) for s in self.syms] + [ys, xs], 1)
        if USE_LN:
            h = F.relu(self.stem_ln(self.obs_stem(oin) + self.mask_stem(mviews)))
            h = F.relu(self.n1(h + self.r1(h)))
            h = F.relu(self.n2(h + self.r2(h)))
        else:
            h = self.stem(torch.cat([oin, mviews], 1))
            h = F.relu(h + self.r1(h))
            h = F.relu(h + self.r2(h))
        a = F.relu(self.h3(h) + self.h1(h))
        o = self.hout(a)
        place = None
        for i, l in enumerate(self.lays):
            m = inv_lay(o[:, i * K:(i + 1) * K], l).reshape(B, K * hw)
            place = m if place is None else place + m
        place = place[:, :self.n_place]
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
