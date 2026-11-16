import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
OBS_SYM = False
USE_LN = False
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64
MASK_EMB = 8


def lay(t, tr, fy, fx):
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def unlay(t, tr, fy, fx):
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def norm(c):
    if USE_LN:
        return ChanLN(c)
    return nn.Identity()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        square = self.Y == self.X
        lays = [l for l in LAYOUTS if square or not l[0]]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.layouts = lays
        self.grid = (self.n_place % cells == 0)
        if OBS_SYM:
            syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1)]
            self.syms = [s for s in syms if square or not s[0]]
        else:
            self.syms = [(0, 0, 0)]
        obs_in = int(spec.in_channels) * len(self.syms) + 2
        if self.grid:
            self.K = self.n_place // cells
            mask_in = self.K * len(self.layouts)
        else:
            self.K = 1
            self.mask_embed = nn.Linear(self.n_place, cells * MASK_EMB)
            mask_in = MASK_EMB
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(obs_in, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(mask_in, STEM, 1), ChanLN(STEM), nn.ReLU())
        else:
            self.stem1 = nn.Conv2d(obs_in + mask_in, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.n0 = norm(WIDTH)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n1 = norm(WIDTH)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = norm(WIDTH)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid:
            self.out = nn.Conv2d(HEAD, self.K * len(self.layouts), 1)
        else:
            self.out = nn.Conv2d(HEAD, MASK_EMB, 1)
            self.place_lin = nn.Linear(MASK_EMB * cells, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if USE_LN:
            for m in [self.out, self.done_head, self.value_head]:
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = torch.cat([lay(obs, *s) for s in self.syms] + [ys, xs], dim=1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.grid:
            m4 = mp.view(B, self.K, Y, X)
            mi = torch.cat([lay(m4, *l) for l in self.layouts], dim=1)
        else:
            mi = self.mask_embed(mp).view(B, MASK_EMB, Y, X)
        if USE_LN:
            s = self.obs_stem(o) + self.mask_stem(mi)
        else:
            s = F.relu(self.stem1(torch.cat([o, mi], dim=1)))
        x = F.relu(self.n0(self.stem2(s)))
        x = F.relu(self.n1(x + self.r1(x)))
        x = F.relu(self.n2(x + self.r2(x)))
        h = F.relu(self.h3(x) + self.h1(x))
        out = self.out(h)
        if self.grid:
            place = None
            for i, l in enumerate(self.layouts):
                part = unlay(out[:, i * self.K:(i + 1) * self.K], *l).reshape(B, -1)
                place = part if place is None else place + part
        else:
            place = self.place_lin(out.flatten(1))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
