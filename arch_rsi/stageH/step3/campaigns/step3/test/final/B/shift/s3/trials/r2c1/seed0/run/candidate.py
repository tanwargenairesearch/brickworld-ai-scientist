import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

# (transpose, yflip, xflip)
LAYOUTS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]
TYPE_MINOR = False
OBS_SYM = True
USE_LN = True
OUT_GAIN = 0.01
GLOBAL_CTX = True
DILATED = True
DENSE = False

ALL8 = [(False, False, False), (False, False, True), (False, True, False), (False, True, True),
        (True, False, False), (True, False, True), (True, True, False), (True, True, True)]

STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
MEMB = 8
DC = 16


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
        self.n_actions = int(spec.n_actions)
        K = self.n_place // (Y * X)
        self.K = K
        self.grid_ok = (not DENSE) and K >= 1 and K * Y * X == self.n_place
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
            self.mask_lin = nn.Linear(self.n_actions, MEMB * Y * X)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), CLN(STEM), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
            self.mask_stem = nn.Sequential(nn.Conv2d(c_mask, STEM, 1), CLN(STEM), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
            self.stem_norm = CLN(WIDTH)
        else:
            self.stem = nn.Sequential(nn.Conv2d(c_obs + c_mask, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        dils = [1, 2, 1] if DILATED else [1, 1]
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in dils])
        self.norms = nn.ModuleList([CLN(WIDTH) for _ in dils] if USE_LN else [])
        self.ctx = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in dils] if GLOBAL_CTX else [])
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid_ok:
            self.hout = nn.Conv2d(HEAD, K * len(self.layouts), 1)
            outs = [self.hout]
        elif DENSE:
            self.dproj = nn.Conv2d(HEAD, DC, 1)
            self.dlin = nn.Linear(DC * Y * X, self.n_actions)
            outs = [self.dlin]
        else:
            self.hout = nn.Conv2d(HEAD, 4, 1)
            self.hlin = nn.Linear(4 * Y * X, self.n_place)
            outs = [self.hlin]
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        if OUT_GAIN > 0:
            for m in outs + [self.done_head, self.value_head]:
                nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        views = [apply_layout(obs, v) for v in self.views]
        o = torch.cat(views + [ys, xs], dim=1)
        mf = mask.to(obs.dtype)
        if self.grid_ok:
            mp = mf[:, :self.n_place]
            if TYPE_MINOR:
                mg = mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
            else:
                mg = mp.reshape(B, K, Y, X)
            m = torch.cat([apply_layout(mg, l) for l in self.layouts], dim=1)
        else:
            m = self.mask_lin(mf).view(B, MEMB, Y, X)
        if USE_LN:
            x = F.relu(self.stem_norm(self.obs_stem(o) + self.mask_stem(m)))
        else:
            x = self.stem(torch.cat([o, m], dim=1))
        for i, conv in enumerate(self.convs):
            y = x + conv(x)
            if GLOBAL_CTX:
                gx = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
                y = y + self.ctx[i](gx).unsqueeze(-1).unsqueeze(-1)
            if USE_LN:
                y = self.norms[i](y)
            x = F.relu(y)
        f = x
        h = F.relu(self.h3(f) + self.h1(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g)
        if self.grid_ok:
            out = self.hout(h)
            place = None
            for i, l in enumerate(self.layouts):
                oi = invert_layout(out[:, i * K:(i + 1) * K], l)
                if TYPE_MINOR:
                    oi = oi.permute(0, 2, 3, 1)
                oi = oi.reshape(B, -1)
                place = oi if place is None else place + oi
            logits = torch.cat([place, done], dim=1)
        elif DENSE:
            d = F.relu(self.dproj(h)).flatten(1)
            logits = self.dlin(d) + F.pad(done, (self.n_actions - 1, 0))
        else:
            place = self.hlin(self.hout(h).flatten(1))
            logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
