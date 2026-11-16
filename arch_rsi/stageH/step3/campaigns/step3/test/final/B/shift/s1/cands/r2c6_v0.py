import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = ((0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1))
TYPE_MINOR = True
OBS_SYMS = True
USE_LN = True
GCTX = True
DILATED = True
OUT_GAIN = 0.01
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


def fwd_layout(t, lay):
    tr, fy, fx = lay
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def inv_layout(t, lay):
    tr, fy, fx = lay
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


class Block(nn.Module):
    def __init__(self, c, dil, ln, gctx):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.ln = ChanLN(c) if ln else None
        self.g = nn.Linear(2 * c, c) if gctx else None

    def forward(self, x):
        h = self.conv(x)
        if self.g is not None:
            h = h + self.g(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))[:, :, None, None]
        if self.ln is not None:
            h = self.ln(h)
        return F.relu(x + h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        hw = self.Y * self.X
        self.K = (self.n_place + hw - 1) // hw
        lays = [l for l in LAYOUTS if self.Y == self.X or l[0] == 0]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.lays = lays
        syms = [(0, 0, 0)]
        if OBS_SYMS:
            syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if self.Y == self.X or t == 0]
        self.syms = syms
        c_obs = spec.in_channels * len(syms) + 2
        c_mask = self.K * len(lays)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(c_mask, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(STEM, WIDTH, 1), ChanLN(WIDTH), nn.ReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(c_obs + c_mask, STEM, 1), nn.ReLU(),
                                      nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        dils = [1, 2, 1] if DILATED else [1, 1]
        self.blocks = nn.ModuleList([Block(WIDTH, d, USE_LN, GCTX) for d in dils])
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.out = nn.Conv2d(HEAD, self.K * len(lays), 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        if OUT_GAIN > 0:
            for m in (self.out, self.done_head, self.value_head):
                nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
                nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = torch.cat([fwd_layout(obs, s) for s in self.syms] + [ys, xs], 1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        pad = K * Y * X - self.n_place
        if pad > 0:
            mp = F.pad(mp, (0, pad))
        if TYPE_MINOR:
            mp = mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
        else:
            mp = mp.reshape(B, K, Y, X)
        m = torch.cat([fwd_layout(mp, l) for l in self.lays], 1)
        if USE_LN:
            x = self.stem2(self.obs_stem(o) + self.mask_stem(m))
        else:
            x = self.stem(torch.cat([o, m], 1))
        for blk in self.blocks:
            x = blk(x)
        h = F.relu(self.h3(x) + self.h1(x))
        maps = self.out(h)
        place = 0
        for i, l in enumerate(self.lays):
            mi = inv_layout(maps[:, i * K:(i + 1) * K], l)
            if TYPE_MINOR:
                mi = mi.permute(0, 2, 3, 1)
            place = place + mi.reshape(B, -1)
        place = place[:, :self.n_place]
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
