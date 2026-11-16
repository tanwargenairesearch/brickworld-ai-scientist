import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
OBS_SYM = True
USE_LN = True
OUT_GAIN = 0.01

STEM = 64
WIDTH = 40
HEADC = 32
GH = 64


def fwd_t(v, t, fy, fx):
    if t:
        v = v.transpose(-1, -2)
    if fy:
        v = v.flip(-2)
    if fx:
        v = v.flip(-1)
    return v


def inv_t(v, t, fy, fx):
    if fx:
        v = v.flip(-1)
    if fy:
        v = v.flip(-2)
    if t:
        v = v.transpose(-1, -2)
    return v


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def norm(c):
    return ChanLN(c) if USE_LN else nn.Identity()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        self.K = (self.n_place + cells - 1) // cells
        sq = self.Y == self.X
        self.layouts = [l for l in LAYOUTS if sq or l[0] == 0]
        if OBS_SYM:
            self.syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or t == 0]
        else:
            self.syms = [(0, 0, 0)]
        c_obs = int(spec.in_channels) * len(self.syms) + 2
        c_mask = self.K * len(self.layouts)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(c_mask, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(2 * STEM, WIDTH, 1), ChanLN(WIDTH), nn.ReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(c_obs + c_mask, STEM, 1), nn.ReLU(),
                                      nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n1 = norm(WIDTH)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = norm(WIDTH)
        self.h3 = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADC, 1)
        self.hout = nn.Conv2d(HEADC, self.K * len(self.layouts), 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        if OUT_GAIN > 0:
            for m in (self.hout, self.done_head, self.value_head):
                nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        cells = Y * X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        ob = torch.cat([fwd_t(obs, *s) for s in self.syms] + [ys, xs], dim=1)
        m = mask[:, :self.n_place].to(obs.dtype)
        if K * cells > self.n_place:
            m = F.pad(m, (0, K * cells - self.n_place))
        m = m.reshape(B, K, Y, X)
        mk = torch.cat([fwd_t(m, *l) for l in self.layouts], dim=1)
        if USE_LN:
            f = self.stem2(torch.cat([self.obs_stem(ob), self.mask_stem(mk)], dim=1))
        else:
            f = self.stem(torch.cat([ob, mk], dim=1))
        f = F.relu(self.n1(f + self.r1(f)))
        f = F.relu(self.n2(f + self.r2(f)))
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        place = None
        for i, l in enumerate(self.layouts):
            p = inv_t(out[:, i * K:(i + 1) * K], *l).reshape(B, K * cells)[:, :self.n_place]
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
