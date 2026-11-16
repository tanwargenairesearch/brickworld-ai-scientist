import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

# (transpose, yflip, xflip)
LAYOUTS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]
OBS_SYM = False
USE_LN = True
OUT_GAIN = 0.01

ALL8 = [(False, False, False), (False, False, True), (False, True, False), (False, True, True),
        (True, False, False), (True, False, True), (True, True, False), (True, True, True)]

STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
MEMB = 8


def apply_layout(t, lay):
    tr, fy, fx = lay
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def invert_layout(t, lay):
    tr, fy, fx = lay
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.Y, self.X = Y, X
        self.n_place = int(spec.n_place)
        K = self.n_place // (Y * X)
        self.K = K
        self.grid_ok = K >= 1 and K * Y * X == self.n_place
        sq = Y == X
        lays = [l for l in LAYOUTS if sq or not l[0]]
        if len(lays) == 0:
            lays = [(False, False, False)]
        self.layouts = lays
        if OBS_SYM:
            self.views = [l for l in ALL8 if sq or not l[0]]
        else:
            self.views = [(False, False, False)]
        c_obs = int(spec.in_channels) * len(self.views) + 2
        if self.grid_ok:
            c_mask = K * len(self.layouts)
        else:
            c_mask = MEMB
            self.mask_lin = nn.Linear(self.n_place, MEMB * Y * X)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), CLN(STEM), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
            self.mask_stem = nn.Sequential(nn.Conv2d(c_mask, STEM, 1), CLN(STEM), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
            self.stem_norm = CLN(WIDTH)
            self.n1 = CLN(WIDTH)
            self.n2 = CLN(WIDTH)
        else:
            self.stem = nn.Sequential(nn.Conv2d(c_obs + c_mask, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid_ok:
            self.hout = nn.Conv2d(HEAD, K * len(self.layouts), 1)
        else:
            self.hout = nn.Conv2d(HEAD, 4, 1)
            self.hlin = nn.Linear(4 * Y * X, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        if OUT_GAIN > 0:
            outs = [self.hout, self.done_head, self.value_head]
            if not self.grid_ok:
                outs.append(self.hlin)
            for m in outs:
                nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        views = [apply_layout(obs, v) for v in self.views]
        o = torch.cat(views + [ys, xs], dim=1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.grid_ok:
            mg = mp.reshape(B, self.K, Y, X)
            m = torch.cat([apply_layout(mg, l) for l in self.layouts], dim=1)
        else:
            m = self.mask_lin(mp).view(B, MEMB, Y, X)
        if USE_LN:
            x = F.relu(self.stem_norm(self.obs_stem(o) + self.mask_stem(m)))
            x = F.relu(self.n1(x + self.r1(x)))
            f = F.relu(self.n2(x + self.r2(x)))
        else:
            x = self.stem(torch.cat([o, m], dim=1))
            x = F.relu(x + self.r1(x))
            f = F.relu(x + self.r2(x))
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        if self.grid_ok:
            place = None
            for i, l in enumerate(self.layouts):
                mi = invert_layout(out[:, i * self.K:(i + 1) * self.K], l).reshape(B, -1)
                place = mi if place is None else place + mi
        else:
            place = self.hlin(out.flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
