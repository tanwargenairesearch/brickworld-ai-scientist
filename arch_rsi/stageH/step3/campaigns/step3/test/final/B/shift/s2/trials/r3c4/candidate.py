import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
DONE_FIRST = False
OBS_VIEWS = ["CYX", "YXC", "YCX"]
ACT_ORDER = "KYX"
STEM = 64
WIDTH = 40
HEADW = 32


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


def obs_view(obs, v, C, Y, X):
    B = obs.shape[0]
    if v == "YXC":
        return obs.reshape(B, Y, X, C).permute(0, 3, 1, 2)
    if v == "YCX":
        return obs.reshape(B, Y, C, X).permute(0, 2, 1, 3)
    return obs


def mask_to_grid(mp, B, K, Y, X):
    if ACT_ORDER == "YXK":
        return mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
    if ACT_ORDER == "YKX":
        return mp.reshape(B, Y, K, X).permute(0, 2, 1, 3)
    return mp.reshape(B, K, Y, X)


def grid_to_flat(m, B):
    if ACT_ORDER == "YXK":
        return m.permute(0, 2, 3, 1).reshape(B, -1)
    if ACT_ORDER == "YKX":
        return m.permute(0, 2, 1, 3).reshape(B, -1)
    return m.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        hw = self.Y * self.X
        self.K = (self.n_place + hw - 1) // hw
        sq = self.Y == self.X
        self.lays = [l for l in LAYOUTS if sq or l[0] == 0]
        if len(self.lays) == 0:
            self.lays = [(0, 1, 1)]
        L = len(self.lays)
        cin = self.C * len(OBS_VIEWS) + 2 + self.K * L
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADW, 1)
        self.hout = nn.Conv2d(HEADW, self.K * L, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K, C = self.Y, self.X, self.K, self.C
        hw = Y * X
        mf = mask.to(obs.dtype)
        if DONE_FIRST:
            mp = mf[:, 1:1 + self.n_place]
        else:
            mp = mf[:, :self.n_place]
        if K * hw > self.n_place:
            mp = F.pad(mp, (0, K * hw - self.n_place))
        mg = mask_to_grid(mp, B, K, Y, X)
        mviews = torch.cat([fwd_lay(mg, l) for l in self.lays], 1)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        oin = torch.cat([obs_view(obs, v, C, Y, X) for v in OBS_VIEWS] + [ys, xs], 1)
        h = self.stem(torch.cat([oin, mviews], 1))
        h = F.relu(h + self.r1(h))
        h = F.relu(h + self.r2(h))
        a = F.relu(self.h3(h) + self.h1(h))
        o = self.hout(a)
        place = None
        for i, l in enumerate(self.lays):
            m = grid_to_flat(inv_lay(o[:, i * K:(i + 1) * K], l), B)
            place = m if place is None else place + m
        place = place[:, :self.n_place]
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        done = self.done_head(g)
        if DONE_FIRST:
            logits = torch.cat([done, place], 1)
        else:
            logits = torch.cat([place, done], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
