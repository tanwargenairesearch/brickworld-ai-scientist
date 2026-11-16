import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [(0, 1, 0)]
OBS_SYM = False
GCTX = False
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


def to_obs(m, t, fy, fx):
    if t:
        m = m.transpose(-1, -2)
    if fy:
        m = m.flip(-2)
    if fx:
        m = m.flip(-1)
    return m


def to_act(m, t, fy, fx):
    if fx:
        m = m.flip(-1)
    if fy:
        m = m.flip(-2)
    if t:
        m = m.transpose(-1, -2)
    return m


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = spec.Y, spec.X
        self.Y, self.X = Y, X
        self.n_place = spec.n_place
        sq = Y == X
        lays = [l for l in LAYOUTS if sq or l[0] == 0]
        self.layouts = lays if len(lays) > 0 else [(0, 0, 0)]
        if OBS_SYM:
            self.osyms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or t == 0]
        else:
            self.osyms = [(0, 0, 0)]
        K = spec.n_place // (Y * X)
        self.direct = K > 0 and K * Y * X == spec.n_place
        if self.direct:
            self.K = K
        else:
            self.K = 8
            self.mask_embed = nn.Linear(spec.n_place, self.K * Y * X)
            self.out_embed = nn.Linear(self.K * Y * X, spec.n_place)
        nL = len(self.layouts)
        cin_obs = spec.in_channels * len(self.osyms) + 2
        cin_mask = self.K * nL
        self.stem = nn.Sequential(nn.Conv2d(cin_obs + cin_mask, STEM, 1), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = Res(WIDTH)
        self.r2 = Res(WIDTH)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if GCTX:
            self.ctx = nn.Linear(2 * WIDTH, HEAD)
        self.out = nn.Conv2d(HEAD, self.K * nL, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        views = [to_obs(obs, *s) for s in self.osyms]
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = torch.cat(views + [yy, xx], 1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.direct:
            m = mp.reshape(B, self.K, Y, X)
        else:
            m = self.mask_embed(mp).reshape(B, self.K, Y, X)
        ms = torch.cat([to_obs(m, *l) for l in self.layouts], 1)
        s = self.stem(torch.cat([o, ms], 1))
        f = self.proj(s)
        f = self.r1(f)
        f = self.r2(f)
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        hp = self.h3(f) + self.h1(f)
        if GCTX:
            hp = hp + self.ctx(pooled)[:, :, None, None]
        h = F.relu(hp)
        maps = self.out(h)
        K = self.K
        place = 0
        for i, l in enumerate(self.layouts):
            place = place + to_act(maps[:, i * K:(i + 1) * K], *l)
        place = place.reshape(B, K * Y * X)
        if not self.direct:
            place = self.out_embed(place)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
