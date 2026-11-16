import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
TYPE_MINOR = True
OBS_SYM = True
USE_LN = True
GCTX = True
DILATE = True
DENSE = False
STEM = 64
MSTEM = 32
WIDTH = 40
HEADW = 32
GLOB = 64


def lay_fwd(t, l):
    tr, fy, fx = l
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def lay_inv(t, l):
    tr, fy, fx = l
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


def obs_views(obs, square):
    vs = []
    for tr in ([0, 1] if square else [0]):
        for fy in (0, 1):
            for fx in (0, 1):
                vs.append(lay_fwd(obs, (tr, fy, fx)))
    return torch.cat(vs, 1)


class CellLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c, dil, use_ln, gctx):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.use_ln = use_ln
        self.gctx = gctx
        if use_ln:
            self.ln = CellLN(c)
        if gctx:
            self.g = nn.Linear(2 * c, c)

    def forward(self, x):
        y = x + self.conv(x)
        if self.gctx:
            y = y + self.g(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))[:, :, None, None]
        if self.use_ln:
            y = self.ln(y)
        return F.relu(y)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        yx = self.Y * self.X
        self.K = (self.n_place + yx - 1) // yx
        self.padn = self.K * yx - self.n_place
        self.square = self.Y == self.X
        self.layouts = [l for l in LAYOUTS if self.square or l[0] == 0]
        nL = len(self.layouts)
        nv = ((8 if self.square else 4) if OBS_SYM else 1)
        c_obs = int(spec.in_channels) * nv + 2
        if DENSE:
            self.me = max(2, min(8, 150000 // (self.n_place * yx)))
            self.mask_lin = nn.Linear(self.n_place, yx * self.me)
            c_mask = self.me
        else:
            c_mask = self.K * nL
        if USE_LN:
            self.obs_stem = nn.Conv2d(c_obs, STEM, 1)
            self.obs_ln = CellLN(STEM)
            self.mask_stem = nn.Conv2d(c_mask, MSTEM, 1)
            self.mask_ln = CellLN(MSTEM)
            self.merge = nn.Conv2d(STEM + MSTEM, WIDTH, 1)
            self.merge_ln = CellLN(WIDTH)
        else:
            self.stem1 = nn.Conv2d(c_obs + c_mask, STEM, 1)
            self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.blocks = nn.ModuleList([Block(WIDTH, 2 if (DILATE and i == 1) else 1, USE_LN, GCTX) for i in range(2)])
        self.h3 = nn.Conv2d(WIDTH, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADW, 1)
        if DENSE:
            self.dc = max(2, min(16, 300000 // (yx * self.n_place)))
            self.dproj = nn.Conv2d(HEADW, self.dc, 1)
            self.dout = nn.Linear(yx * self.dc, self.n_place)
            outs = [self.dout]
        else:
            self.out = nn.Conv2d(HEADW, self.K * nL, 1)
            outs = [self.out]
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if USE_LN:
            for m in outs + [self.done_head, self.value_head]:
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        yx = Y * X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = obs_views(obs, self.square) if OBS_SYM else obs
        o = torch.cat([o, ys, xs], 1)
        m = mask[:, :self.n_place].to(obs.dtype)
        if DENSE:
            mv = self.mask_lin(m).view(B, self.me, Y, X)
        else:
            if self.padn > 0:
                m = F.pad(m, (0, self.padn))
            if TYPE_MINOR:
                m = m.view(B, Y, X, K).permute(0, 3, 1, 2)
            else:
                m = m.view(B, K, Y, X)
            mv = torch.cat([lay_fwd(m, l) for l in self.layouts], 1)
        if USE_LN:
            a = F.relu(self.obs_ln(self.obs_stem(o)))
            b = F.relu(self.mask_ln(self.mask_stem(mv)))
            x = F.relu(self.merge_ln(self.merge(torch.cat([a, b], 1))))
        else:
            x = F.relu(self.stem1(torch.cat([o, mv], 1)))
            x = F.relu(self.stem2(x))
        for blk in self.blocks:
            x = blk(x)
        h = F.relu(self.h3(x) + self.h1(x))
        if DENSE:
            place = self.dout(F.relu(self.dproj(h)).flatten(1))
        else:
            maps = self.out(h)
            place = 0
            for i, l in enumerate(self.layouts):
                place = place + lay_inv(maps[:, i * K:(i + 1) * K], l)
            if TYPE_MINOR:
                place = place.permute(0, 2, 3, 1)
            place = place.reshape(B, K * yx)[:, :self.n_place]
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
