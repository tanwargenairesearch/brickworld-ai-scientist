import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [("YXK", 0, 0, 0), ("YXK", 1, 0, 0), ("YXK", 0, 1, 0), ("YXK", 0, 0, 1)]
OBS_SYM = 1
USE_LN = 0
COMBO = 0
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


def read_order(m, order, K, Y, X):
    B = m.shape[0]
    if order == "YXK":
        return m.reshape(B, Y, X, K).permute(0, 3, 1, 2)
    if order == "YKX":
        return m.reshape(B, Y, K, X).permute(0, 2, 1, 3)
    return m.reshape(B, K, Y, X)


def write_order(o, order):
    B = o.shape[0]
    if order == "YXK":
        o = o.permute(0, 2, 3, 1)
    elif order == "YKX":
        o = o.permute(0, 2, 1, 3)
    return o.reshape(B, -1)


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
            self.norm = CellLN(c)
        if gctx:
            self.g = nn.Linear(c, c)

    def forward(self, x):
        h = self.norm(x) if self.use_ln else x
        y = self.conv(h)
        if self.gctx:
            y = y + self.g(h.mean((2, 3)))[:, :, None, None]
        if self.use_ln:
            return x + F.relu(y)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cells = self.Y * self.X
        self.K = (self.n_place + cells - 1) // cells
        sq = self.Y == self.X
        self.layouts = [l for l in LAYOUTS if sq or not l[1]]
        if len(self.layouts) == 0:
            self.layouts = [(LAYOUTS[0][0], 0, 1, 1)]
        if OBS_SYM:
            self.syms = [(t, fy, fx) for t in (0, 1) for fy in (0, 1) for fx in (0, 1) if sq or not t]
        else:
            self.syms = [(0, 0, 0)]
        n_obs = int(spec.in_channels) * len(self.syms)
        n_mask = self.K * len(self.layouts)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(n_obs + 2, STEM, 1), CellLN(STEM), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1))
            self.mask_stem = nn.Sequential(nn.Conv2d(n_mask + 2, MSTEM, 1), CellLN(MSTEM), nn.ReLU(), nn.Conv2d(MSTEM, WIDTH, 1))
            self.n0 = CellLN(WIDTH)
            self.nf = CellLN(WIDTH)
        else:
            self.stem = nn.Sequential(nn.Conv2d(n_obs + n_mask + 2, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        blocks = [Block(WIDTH, 1, USE_LN, COMBO), Block(WIDTH, 1, USE_LN, COMBO)]
        if COMBO:
            blocks.append(Block(WIDTH, 2, USE_LN, COMBO))
        self.blocks = nn.ModuleList(blocks)
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
        mm = torch.cat([fwd_layout(read_order(m, l[0], K, Y, X), l[1], l[2], l[3]) for l in self.layouts], dim=1)
        if USE_LN:
            x = self.n0(self.obs_stem(torch.cat([o, coords], dim=1)) + self.mask_stem(torch.cat([mm, coords], dim=1)))
        else:
            x = self.stem(torch.cat([o, mm, coords], dim=1))
        for b in self.blocks:
            x = b(x)
        f = F.relu(self.nf(x)) if USE_LN else x
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        place = None
        for i, l in enumerate(self.layouts):
            p = write_order(inv_layout(out[:, i * K:(i + 1) * K], l[1], l[2], l[3]), l[0])[:, :self.n_place]
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
