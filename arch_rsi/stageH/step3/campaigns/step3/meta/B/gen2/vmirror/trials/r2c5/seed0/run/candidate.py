import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
GCTX = True

STEM = 64
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
        lays = [l for l in LAYOUTS if sq or l[0] == 0]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.lays = lays
        self.syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or t == 0]
        nl = len(self.lays)
        c_in = spec.in_channels * len(self.syms) + 2 + self.K * nl
        self.stem = nn.Sequential(nn.Conv2d(c_in, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, self.K * nl, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        if GCTX:
            self.gctx = nn.Linear(GLOB, HEAD)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

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
        f = self.stem(torch.cat(views + [coords] + mv, dim=1))
        f = F.relu(f + self.r1(f))
        f = F.relu(f + self.r2(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        hpre = self.h3(f) + self.h1(f)
        if GCTX:
            hpre = hpre + self.gctx(g)[:, :, None, None]
        h = F.relu(hpre)
        o = self.hout(h)
        place = None
        for i, l in enumerate(self.lays):
            p = fwd(o[:, i * K:(i + 1) * K], l).reshape(B, -1)
            place = p if place is None else place + p
        place = place[:, :self.n_place]
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
