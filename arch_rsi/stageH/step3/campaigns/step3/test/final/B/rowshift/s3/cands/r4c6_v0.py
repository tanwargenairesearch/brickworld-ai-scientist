import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ["compact"]
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
OBS_SYM = 1
USE_LN = 1
COMBO = 1
GALL = 1
DONE_FIRST = 0
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
        self.n_actions = int(spec.n_actions)
        cells = self.Y * self.X
        self.Kstd = (self.n_place + cells - 1) // cells
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.cdims = [(max(self.Y - dy + 1, 1), max(self.X - dx + 1, 1)) for dy, dx in self.fps]
        compact_ok = len(self.fps) > 0 and sum(a * b for a, b in self.cdims) == self.n_place
        modes = [md for md in MODES if md != "compact" or compact_ok]
        if len(modes) == 0:
            modes = ["std"]
        sq = self.Y == self.X
        lays = [l for l in LAYOUTS if sq or not l[0]]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.comps = []
        for md in modes:
            k = len(self.fps) if md == "compact" else self.Kstd
            for l in lays:
                self.comps.append((md, l, k))
        n_mask = sum(c[2] for c in self.comps)
        if OBS_SYM:
            self.syms = [(t, fy, fx) for t in (0, 1) for fy in (0, 1) for fx in (0, 1) if sq or not t]
        else:
            self.syms = [(0, 0, 0)]
        n_obs = int(spec.in_channels) * len(self.syms)
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
        outs = [self.hout, self.done_head, self.value_head]
        if GALL:
            self.gall = nn.Linear(GLOB, self.n_actions)
            outs.append(self.gall)
        if USE_LN:
            for m in outs:
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)

    def read(self, mf, md):
        B = mf.shape[0]
        Y, X = self.Y, self.X
        if md == "compact":
            segs = []
            o = 0
            for yc, xc in self.cdims:
                sz = yc * xc
                seg = mf[:, o:o + sz].reshape(B, 1, yc, xc)
                segs.append(F.pad(seg, (0, X - xc, 0, Y - yc)))
                o += sz
            return torch.cat(segs, dim=1)
        st = 1 if md == "off1" else 0
        m = mf[:, st:st + self.n_place]
        pad = self.Kstd * Y * X - self.n_place
        if pad > 0:
            m = F.pad(m, (0, pad))
        return m.reshape(B, self.Kstd, Y, X)

    def write(self, o, md):
        B = o.shape[0]
        if md == "compact":
            parts = []
            for t, (yc, xc) in enumerate(self.cdims):
                parts.append(o[:, t, :yc, :xc].reshape(B, -1))
            return F.pad(torch.cat(parts, dim=1), (0, 1))
        p = o.reshape(B, -1)[:, :self.n_place]
        if md == "off1":
            return F.pad(p, (1, 0))
        return F.pad(p, (0, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        coords = torch.cat([yy, xx], dim=1)
        o = torch.cat([fwd_layout(obs, *s) for s in self.syms], dim=1)
        mf = mask.to(obs.dtype)
        mm = torch.cat([fwd_layout(self.read(mf, c[0]), *c[1]) for c in self.comps], dim=1)
        if USE_LN:
            x = self.n0(self.obs_stem(torch.cat([o, coords], dim=1)) + self.mask_stem(torch.cat([mm, coords], dim=1)))
        else:
            x = self.stem(torch.cat([o, mm, coords], dim=1))
        for b in self.blocks:
            x = b(x)
        f = F.relu(self.nf(x)) if USE_LN else x
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        logits = None
        c0 = 0
        for md, l, k in self.comps:
            p = self.write(inv_layout(out[:, c0:c0 + k], *l), md)
            c0 += k
            logits = p if logits is None else logits + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        if DONE_FIRST:
            d = F.pad(d, (0, self.n_actions - 1))
        else:
            d = F.pad(d, (self.n_actions - 1, 0))
        logits = logits + d
        if GALL:
            logits = logits + self.gall(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
