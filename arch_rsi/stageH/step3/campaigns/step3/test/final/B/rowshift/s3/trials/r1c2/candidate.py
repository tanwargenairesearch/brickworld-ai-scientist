import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [(0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
OBS_SYM = 0
USE_LN = 0
STEM = 64
MSTEM = 32
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
            self.layouts = [(0, 1, 1)]
        if OBS_SYM:
            self.syms = [(t, fy, fx) for t in (0, 1) for fy in (0, 1) for fx in (0, 1) if sq or not t]
        else:
            self.syms = [(0, 0, 0)]
        n_obs = int(spec.in_channels) * len(self.syms)
        n_mask = self.K * len(self.layouts)
        self.n_mask = n_mask
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(n_obs + 2, STEM, 1), CellLN(STEM), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
            self.mask_stem = nn.Sequential(nn.Conv2d(n_mask + 2, MSTEM, 1), CellLN(MSTEM), nn.ReLU(), nn.Conv2d(MSTEM, WIDTH, 1))
            self.n0 = CellLN(WIDTH)
            self.n1 = CellLN(WIDTH)
            self.n2 = CellLN(WIDTH)
            self.nf = CellLN(WIDTH)
        else:
            self.stem = nn.Sequential(nn.Conv2d(n_obs + n_mask + 2, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADW, 1)
        self.hout = nn.Conv2d(HEADW, n_mask, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if USE_LN:
            for m in (self.hout, self.done_head, self.value_head):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        coords = torch.cat([yy, xx], dim=1)
        o = torch.cat([fwd_layout(obs, *s) for s in self.syms], dim=1)
        m = mask[:, :self.n_place].to(obs.dtype)
        pad = K * Y * X - self.n_place
        if pad > 0:
            m = F.pad(m, (0, pad))
        m = m.reshape(B, K, Y, X)
        mm = torch.cat([fwd_layout(m, *l) for l in self.layouts], dim=1)
        if USE_LN:
            s = self.n0(self.obs_stem(torch.cat([o, coords], dim=1)) + self.mask_stem(torch.cat([mm, coords], dim=1)))
            x = s + F.relu(self.r1(self.n1(s)))
            x = x + F.relu(self.r2(self.n2(x)))
            f = F.relu(self.nf(x))
        else:
            s = self.stem(torch.cat([o, mm, coords], dim=1))
            x = F.relu(s + self.r1(s))
            f = F.relu(x + self.r2(x))
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        place = None
        for i, l in enumerate(self.layouts):
            p = inv_layout(out[:, i * K:(i + 1) * K], *l).reshape(B, -1)[:, :self.n_place]
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
