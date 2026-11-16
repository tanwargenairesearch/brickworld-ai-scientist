import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
TYPE_MINOR = True
OBS_SYM = True
USE_LN = True
GCTX = True
DIL = True
GENERIC = False
SEP_CRITIC = False
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64
GEN_CH = 16
CRIT = 32


def lay(g, t, fy, fx):
    if t:
        g = g.transpose(2, 3)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def unlay(g, t, fy, fx):
    if fx:
        g = g.flip(3)
    if fy:
        g = g.flip(2)
    if t:
        g = g.transpose(2, 3)
    return g


class ChanLN(nn.Module):
    def __init__(self, c, on):
        super().__init__()
        self.on = on
        self.ln = nn.LayerNorm(c) if on else nn.Identity()

    def forward(self, x):
        if not self.on:
            return x
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, w, dil, use_ln, gctx):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=dil, dilation=dil)
        self.ln = ChanLN(w, use_ln)
        self.gctx = gctx
        if gctx:
            self.g = nn.Linear(2 * w, w)

    def forward(self, f):
        h = self.conv(f)
        if self.gctx:
            pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
            h = h + self.g(pooled)[:, :, None, None]
        return F.relu(self.ln(f + h))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ny = int(spec.Y)
        self.nx = int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_extra = int(spec.n_actions) - self.n_place
        cells = self.ny * self.nx
        sq = self.ny == self.nx
        self.grid_mask = ((self.n_place % cells) == 0) and not GENERIC
        self.k = self.n_place // cells
        lays = [l for l in LAYOUTS if sq or l[0] == 0]
        if len(lays) == 0:
            lays = [(0, 0, 0)]
        self.layouts = lays
        allsyms = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)]
        if sq:
            allsyms = allsyms + [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
        self.syms = allsyms if OBS_SYM else [(0, 0, 0)]
        nl = len(self.layouts)
        cin = int(spec.in_channels) * len(self.syms) + 2
        if self.grid_mask:
            mch = self.k * nl
            nout = self.k * nl
        else:
            mch = 8
            nout = GEN_CH
            self.mask_embed = nn.Linear(self.n_place, 8 * cells)
            self.place_out = nn.Linear(GEN_CH * cells, self.n_place)
            nn.init.orthogonal_(self.place_out.weight, 0.01)
            nn.init.zeros_(self.place_out.bias)
        self.mch = mch
        self.obs_stem = nn.Conv2d(cin, STEM, 1)
        self.mask_stem = nn.Conv2d(mch, STEM, 1)
        self.ln_o = ChanLN(STEM, USE_LN)
        self.ln_m = ChanLN(STEM, USE_LN)
        self.col = nn.Conv2d(STEM, WIDTH, 1)
        self.ln_c = ChanLN(WIDTH, USE_LN)
        dils = [1, 1] + ([2] if DIL else [])
        self.blocks = nn.ModuleList([Block(WIDTH, d, USE_LN, GCTX) for d in dils])
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, nout, 1)
        self.glob = nn.Linear(2 * WIDTH, GLOB)
        self.done_head = nn.Linear(GLOB, self.n_extra)
        self.value_head = nn.Linear(GLOB, 1)
        if USE_LN:
            if self.grid_mask:
                nn.init.orthogonal_(self.hout.weight, 0.01)
                nn.init.zeros_(self.hout.bias)
            nn.init.orthogonal_(self.done_head.weight, 0.01)
            nn.init.zeros_(self.done_head.bias)
        if SEP_CRITIC:
            self.c_stem = nn.Conv2d(int(spec.in_channels) + 2, CRIT, 1)
            self.c_ln = ChanLN(CRIT, USE_LN)
            self.c_conv = nn.Conv2d(CRIT, CRIT, 3, padding=1)
            self.c_fc = nn.Linear(2 * CRIT, 64)
            self.c_out = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        ny = self.ny
        nx = self.nx
        views = [lay(obs, t, fy, fx) for (t, fy, fx) in self.syms]
        yy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(b, 1, ny, nx)
        xx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(b, 1, ny, nx)
        xo = torch.cat(views + [yy, xx], dim=1)
        mp = mask[:, :self.n_place].to(obs.dtype)
        if self.grid_mask:
            if TYPE_MINOR:
                mg = mp.reshape(b, ny, nx, self.k).permute(0, 3, 1, 2)
            else:
                mg = mp.reshape(b, self.k, ny, nx)
            xm = torch.cat([lay(mg, t, fy, fx) for (t, fy, fx) in self.layouts], dim=1)
        else:
            xm = self.mask_embed(mp).reshape(b, self.mch, ny, nx)
        so = self.obs_stem(xo)
        sm = self.mask_stem(xm)
        if USE_LN:
            s = F.relu(self.ln_o(so)) + F.relu(self.ln_m(sm))
        else:
            s = F.relu(so + sm)
        f = F.relu(self.ln_c(self.col(s)))
        for blk in self.blocks:
            f = blk(f)
        h = F.relu(self.h3(f) + self.h1(f))
        o = self.hout(h)
        if self.grid_mask:
            place = None
            for i, (t, fy, fx) in enumerate(self.layouts):
                oi = o[:, i * self.k:(i + 1) * self.k]
                c = unlay(oi, t, fy, fx)
                if TYPE_MINOR:
                    c = c.permute(0, 2, 3, 1)
                c = c.reshape(b, -1)
                place = c if place is None else place + c
        else:
            place = self.place_out(o.reshape(b, -1))
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if SEP_CRITIC:
            xc = torch.cat([obs, yy, xx], dim=1)
            c1 = F.relu(self.c_ln(self.c_stem(xc)))
            c2 = F.relu(c1 + self.c_conv(c1))
            gc = F.relu(self.c_fc(torch.cat([c2.mean((2, 3)), c2.amax((2, 3))], dim=1)))
            value = self.c_out(gc).squeeze(1)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
