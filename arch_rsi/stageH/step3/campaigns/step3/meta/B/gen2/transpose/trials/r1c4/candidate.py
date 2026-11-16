import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUT_SET = "B"
OBS_SYM = True
USE_LN = False
STEM = 64
WIDTH = 40
HEADC = 32
GH = 64

SET_A = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
SET_B = ((0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1))
FLIPS = ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1))
ALL8 = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))


def fwd_t(x, l):
    if l[0]:
        x = x.transpose(-1, -2)
    if l[1]:
        x = x.flip(-2)
    if l[2]:
        x = x.flip(-1)
    return x


def inv_t(x, l):
    if l[2]:
        x = x.flip(-1)
    if l[1]:
        x = x.flip(-2)
    if l[0]:
        x = x.transpose(-1, -2)
    return x


def pick_layouts(base, square):
    if square:
        return list(base)
    out = [l for l in base if l[0] == 0]
    for l in FLIPS:
        if len(out) >= 4:
            break
        if l not in out:
            out.append(l)
    return out


class CellLN(nn.Module):
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
        self.K = max(1, -(-self.n_place // (Y * X)))
        self.pad = self.K * Y * X - self.n_place
        square = Y == X
        self.layouts = pick_layouts(SET_A if LAYOUT_SET == "A" else SET_B, square)
        if OBS_SYM:
            self.views = list(ALL8) if square else list(FLIPS)
        else:
            self.views = [(0, 0, 0)]
        nL = len(self.layouts)
        cin_obs = spec.in_channels * len(self.views) + 2
        cin_mask = self.K * nL
        if USE_LN:
            self.obs_stem = nn.Conv2d(cin_obs, STEM, 1)
            self.obs_ln = CellLN(STEM)
            self.mask_stem = nn.Conv2d(cin_mask, STEM, 1)
            self.mask_ln = CellLN(STEM)
        else:
            self.stem = nn.Conv2d(cin_obs + cin_mask, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.ln2 = CellLN(WIDTH) if USE_LN else nn.Identity()
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n1 = CellLN(WIDTH) if USE_LN else nn.Identity()
        self.n2 = CellLN(WIDTH) if USE_LN else nn.Identity()
        self.h3 = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADC, 1)
        self.hout = nn.Conv2d(HEADC, self.K * nL, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        if USE_LN:
            for m in (self.hout, self.done_head, self.value_head):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)
        yy = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer("coords", torch.cat([yy, xx], 1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        views = [fwd_t(obs, v) for v in self.views]
        coords = self.coords.expand(B, -1, -1, -1).to(obs.dtype)
        o = torch.cat(views + [coords], 1)
        m = mask[:, :self.n_place].to(obs.dtype)
        if self.pad > 0:
            m = F.pad(m, (0, self.pad))
        m = m.reshape(B, K, Y, X)
        ms = torch.cat([fwd_t(m, l) for l in self.layouts], 1)
        if USE_LN:
            s = F.relu(self.obs_ln(self.obs_stem(o))) + F.relu(self.mask_ln(self.mask_stem(ms)))
        else:
            s = F.relu(self.stem(torch.cat([o, ms], 1)))
        f = F.relu(self.ln2(self.stem2(s)))
        f = F.relu(f + self.n1(self.r1(f)))
        f = F.relu(f + self.n2(self.r2(f)))
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h).reshape(B, len(self.layouts), K, Y, X)
        place = None
        for i, l in enumerate(self.layouts):
            p = inv_t(out[:, i], l)
            place = p if place is None else place + p
        place = place.reshape(B, -1)[:, :self.n_place]
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
