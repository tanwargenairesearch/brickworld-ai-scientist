import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = ((1, 0, 0),)
GCTX = False
USE_LN = True
STEM = 64
WIDTH = 40
HEADC = 32
GH = 64


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


class CellLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def norm(c):
    return CellLN(c) if USE_LN else nn.Identity()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.Y, self.X = Y, X
        self.n_place = int(spec.n_place)
        self.K = max(1, -(-self.n_place // (Y * X)))
        self.pad = self.K * Y * X - self.n_place
        lays = [tuple(l) for l in LAYOUTS if (Y == X or l[0] == 0)]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.layouts = lays
        nL = len(self.layouts)
        cin_obs = spec.in_channels + 2
        cin_mask = self.K * nL
        if USE_LN:
            self.obs_stem = nn.Conv2d(cin_obs, STEM, 1)
            self.obs_ln = CellLN(STEM)
            self.mask_stem = nn.Conv2d(cin_mask, STEM, 1)
            self.mask_ln = CellLN(STEM)
        else:
            self.stem = nn.Conv2d(cin_obs + cin_mask, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.ln2 = norm(WIDTH)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n1 = norm(WIDTH)
        self.n2 = norm(WIDTH)
        self.h3 = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADC, 1)
        self.gproj = nn.Linear(GH, HEADC) if GCTX else None
        self.hout = nn.Conv2d(HEADC, self.K * nL, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        for m in (self.hout, self.done_head, self.value_head):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.zeros_(m.bias)
        yy = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer("coords", torch.cat([yy, xx], 1).contiguous())

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        coords = self.coords.expand(B, -1, -1, -1).to(obs.dtype)
        o = torch.cat([obs, coords], 1)
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
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        hpre = self.h3(f) + self.h1(f)
        if self.gproj is not None:
            hpre = hpre + self.gproj(g)[:, :, None, None]
        h = F.relu(hpre)
        out = self.hout(h).reshape(B, len(self.layouts), K, Y, X)
        place = None
        for i, l in enumerate(self.layouts):
            p = inv_t(out[:, i], l)
            place = p if place is None else place + p
        place = place.reshape(B, -1)[:, :self.n_place]
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
