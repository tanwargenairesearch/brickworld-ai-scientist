import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [(0, 0, 0)]
OBS_SYM = 0
USE_LN = 0
RELAY = 8
BOT = 128
LINC = 8
SEP_CRITIC = 0
CRIT = 128
STEM = 64
WIDTH = 40
HEADW = 32
GLOB = 64
RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


def fwd_layout(x, t, fy, fx):
    if t:
        x = x.transpose(2, 3)
    if fy:
        x = x.flip(2)
    if fx:
        x = x.flip(3)
    return x


def inv_layout(x, t, fy, fx):
    if fx:
        x = x.flip(3)
    if fy:
        x = x.flip(2)
    if t:
        x = x.transpose(2, 3)
    return x


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
        cells = self.Y * self.X
        self.K = (self.n_place + cells - 1) // cells
        sq = self.Y == self.X
        self.layouts = [l for l in LAYOUTS if sq or not l[0]]
        if len(self.layouts) == 0:
            self.layouts = [(0, 0, 0)]
        if OBS_SYM:
            self.syms = [(t, fy, fx) for t in (0, 1) for fy in (0, 1) for fx in (0, 1) if sq or not t]
        else:
            self.syms = [(0, 0, 0)]
        n_obs = int(spec.in_channels) * len(self.syms)
        n_mask = self.K * len(self.layouts)
        n_flat_obs = int(spec.in_channels) * cells
        if RELAY > 0:
            self.relay_in = nn.Linear(n_flat_obs + self.n_place, BOT)
            self.relay_out = nn.Linear(BOT, RELAY * cells)
            if USE_LN:
                self.relay_ln = nn.LayerNorm(BOT)
        self.stem1 = nn.Conv2d(n_obs + n_mask + 2 + RELAY, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        if USE_LN:
            self.n_s = CellLN(STEM)
            self.n0 = CellLN(WIDTH)
            self.n1 = CellLN(WIDTH)
            self.n2 = CellLN(WIDTH)
            self.nf = CellLN(WIDTH)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADW, 1)
        self.hout = nn.Conv2d(HEADW, n_mask, 1)
        if LINC > 0:
            self.lproj = nn.Conv2d(WIDTH, LINC, 1)
            self.lhead = nn.Linear(LINC * cells, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        if SEP_CRITIC:
            self.critic = nn.Sequential(nn.Linear(n_flat_obs, CRIT), nn.ReLU(), nn.Linear(CRIT, CRIT), nn.ReLU(), nn.Linear(CRIT, 1))
        else:
            self.value_head = nn.Linear(GLOB, 1)
        if USE_LN:
            outs = [self.hout, self.done_head]
            if LINC > 0:
                outs.append(self.lhead)
            for m in outs:
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        coords = torch.cat([yy, xx], dim=1)
        o = torch.cat([fwd_layout(obs, *s) for s in self.syms], dim=1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        pad = K * Y * X - self.n_place
        m = F.pad(mp, (0, pad)) if pad > 0 else mp
        m = m.reshape(B, K, Y, X)
        mm = torch.cat([fwd_layout(m, *l) for l in self.layouts], dim=1)
        parts = [o, mm, coords]
        if RELAY > 0:
            z = self.relay_in(torch.cat([obs.flatten(1), mp], dim=1))
            if USE_LN:
                z = self.relay_ln(z)
            z = F.relu(z)
            parts.append(self.relay_out(z).reshape(B, RELAY, Y, X))
        s = self.stem1(torch.cat(parts, dim=1))
        if USE_LN:
            s = self.n0(self.stem2(F.relu(self.n_s(s))))
            x = s + F.relu(self.r1(self.n1(s)))
            x = x + F.relu(self.r2(self.n2(x)))
            f = F.relu(self.nf(x))
        else:
            s = F.relu(self.stem2(F.relu(s)))
            x = F.relu(s + self.r1(s))
            f = F.relu(x + self.r2(x))
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        place = None
        for i, l in enumerate(self.layouts):
            p = inv_layout(out[:, i * K:(i + 1) * K], *l).reshape(B, -1)[:, :self.n_place]
            place = p if place is None else place + p
        if LINC > 0:
            place = place + self.lhead(F.relu(self.lproj(f)).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if SEP_CRITIC:
            value = self.critic(obs.flatten(1)).squeeze(1)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
