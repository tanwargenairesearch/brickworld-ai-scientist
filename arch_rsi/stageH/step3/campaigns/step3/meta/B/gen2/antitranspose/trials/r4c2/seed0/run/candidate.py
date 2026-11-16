import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 3e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

# (transpose, yflip, xflip)
LAYOUTS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]
OBS_VIEWS = [(False, False, False), (False, True, True), (True, True, False), (True, False, True)]
MASK_VIEWS = [(False, False, False), (False, False, True), (False, True, False), (False, True, True), (True, False, False), (True, False, True), (True, True, False), (True, True, True)]
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64


def fwd_t(x, t):
    tr, fy, fx = t
    if tr:
        x = x.transpose(-2, -1)
    if fy:
        x = x.flip(-2)
    if fx:
        x = x.flip(-1)
    return x


def inv_t(x, t):
    tr, fy, fx = t
    if fx:
        x = x.flip(-1)
    if fy:
        x = x.flip(-2)
    if tr:
        x = x.transpose(-2, -1)
    return x


def keep(ts, sq):
    out = [t for t in ts if sq or not t[0]]
    if len(out) == 0:
        out = [(False, False, False)]
    return out


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.n_place = int(spec.n_place)
        sq = self.Y == self.X
        self.layouts = keep(LAYOUTS, sq)
        self.syms = keep(OBS_VIEWS, sq)
        self.mviews = keep(MASK_VIEWS if len(MASK_VIEWS) > 0 else LAYOUTS, sq)
        cells = self.Y * self.X
        self.K = self.n_place // cells
        self.exact = self.K > 0 and self.K * cells == self.n_place
        L = len(self.layouts)
        cin_obs = spec.in_channels * len(self.syms)
        if self.exact:
            cin_mask = self.K * len(self.mviews)
        else:
            cin_mask = 8
            self.mask_embed = nn.Linear(self.n_place, 8 * cells)
        self.stem = nn.Conv2d(cin_obs + cin_mask + 2, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.exact:
            self.hout = nn.Conv2d(HEAD, self.K * L, 1)
        else:
            self.hout = nn.Conv2d(HEAD, 4, 1)
            self.hlin = nn.Linear(4 * cells, self.n_place)
        self.glob = nn.Linear(2 * WIDTH, GLOB)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        coords = torch.cat([ys, xs], 1)
        ob = torch.cat([fwd_t(obs, s) for s in self.syms], 1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.exact:
            m = mp.reshape(B, self.K, Y, X)
            mk = torch.cat([fwd_t(m, t) for t in self.mviews], 1)
        else:
            mk = self.mask_embed(mp).reshape(B, 8, Y, X)
        s = F.relu(self.stem(torch.cat([ob, mk, coords], 1)))
        h = F.relu(self.stem2(s))
        h = F.relu(h + self.r1(h))
        h = F.relu(h + self.r2(h))
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        e = F.relu(self.h3(h) + self.h1(h))
        o = self.hout(e)
        if self.exact:
            K = self.K
            place = None
            for i, t in enumerate(self.layouts):
                p = inv_t(o[:, i * K:(i + 1) * K], t).reshape(B, -1)
                place = p if place is None else place + p
        else:
            place = self.hlin(o.flatten(1))
        g = F.relu(self.glob(pooled))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
