import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

# layouts as (transpose, yflip, xflip)
LAYOUTS = [(0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
TYPE_MINOR = False
OBS_SYM = True
USE_LN = True
GCTX = True
DILATE = True
SEP_CRITIC = False
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64
CRIT = 32


def fwd_lay(m, t, fy, fx):
    if t:
        m = m.transpose(-1, -2)
    if fy:
        m = m.flip(-2)
    if fx:
        m = m.flip(-1)
    return m


def inv_lay(m, t, fy, fx):
    if fx:
        m = m.flip(-1)
    if fy:
        m = m.flip(-2)
    if t:
        m = m.transpose(-1, -2)
    return m


def pooled(x):
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        sq = self.Y == self.X
        self.lays = [l for l in LAYOUTS if sq or not l[0]]
        if OBS_SYM:
            self.syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or not t]
        else:
            self.syms = [(0, 0, 0)]
        cells = self.Y * self.X
        self.grid = (self.n_place % cells) == 0
        self.K = self.n_place // cells if self.grid else int(spec.n_types)
        nL = len(self.lays)
        obs_c = int(spec.in_channels) * len(self.syms) + 2
        if self.grid:
            mask_c = self.K * nL
        else:
            mask_c = 8
            self.mask_embed = nn.Linear(self.n_place, 8 * cells)
            self.out_lin = nn.Linear(self.K * nL * cells, self.n_place)
        if USE_LN:
            self.obs_stem = nn.Conv2d(obs_c, STEM, 1)
            self.obs_ln = CLN(STEM)
            self.mask_stem = nn.Conv2d(mask_c, STEM, 1)
            self.mask_ln = CLN(STEM)
            self.ln0 = CLN(WIDTH)
            self.ln1 = CLN(WIDTH)
            self.ln2 = CLN(WIDTH)
        else:
            self.stem = nn.Conv2d(obs_c + mask_c, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        d = 2 if DILATE else 1
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d)
        if GCTX:
            self.g1 = nn.Linear(2 * WIDTH, WIDTH)
            self.g2 = nn.Linear(2 * WIDTH, WIDTH)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, self.K * nL, 1)
        self.glob = nn.Linear(2 * WIDTH, GLOB)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if SEP_CRITIC:
            self.v_stem = nn.Conv2d(int(spec.in_channels) + 2, CRIT, 1)
            self.v_conv = nn.Conv2d(CRIT, CRIT, 3, padding=1)
            self.v_fc = nn.Linear(2 * CRIT, GLOB)
        if USE_LN:
            for lin in (self.hout, self.done_head, self.value_head):
                nn.init.orthogonal_(lin.weight, 0.01)
                nn.init.zeros_(lin.bias)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def block(self, x, conv, g, ln):
        y = conv(x)
        if g is not None:
            y = y + g(pooled(x))[:, :, None, None]
        if ln is not None:
            y = ln(y)
        return F.relu(x + y)

    def forward(self, obs, mask):
        B = obs.shape[0]
        cr = self.coords.expand(B, -1, -1, -1)
        views = [fwd_lay(obs, t, a, b) for (t, a, b) in self.syms]
        views.append(cr)
        o = torch.cat(views, 1)
        mp = mask[:, :self.n_place].float()
        if self.grid:
            if TYPE_MINOR:
                m = mp.reshape(B, self.Y, self.X, self.K).permute(0, 3, 1, 2)
            else:
                m = mp.reshape(B, self.K, self.Y, self.X)
            m = torch.cat([fwd_lay(m, t, a, b) for (t, a, b) in self.lays], 1)
        else:
            m = self.mask_embed(mp).reshape(B, 8, self.Y, self.X)
        if USE_LN:
            s = F.relu(self.obs_ln(self.obs_stem(o))) + F.relu(self.mask_ln(self.mask_stem(m)))
            x = F.relu(self.ln0(self.stem2(s)))
            l1, l2 = self.ln1, self.ln2
        else:
            s = F.relu(self.stem(torch.cat([o, m], 1)))
            x = F.relu(self.stem2(s))
            l1, l2 = None, None
        g1 = self.g1 if GCTX else None
        g2 = self.g2 if GCTX else None
        x = self.block(x, self.c1, g1, l1)
        f = self.block(x, self.c2, g2, l2)
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        if self.grid:
            place = None
            for i, (t, a, b) in enumerate(self.lays):
                mi = inv_lay(out[:, i * self.K:(i + 1) * self.K], t, a, b)
                if TYPE_MINOR:
                    mi = mi.permute(0, 2, 3, 1)
                mi = mi.reshape(B, -1)
                place = mi if place is None else place + mi
        else:
            place = self.out_lin(out.reshape(B, -1))
        g = F.relu(self.glob(pooled(f)))
        logits = torch.cat([place, self.done_head(g)], 1)
        if SEP_CRITIC:
            v = F.relu(self.v_stem(torch.cat([obs, cr], 1)))
            v = F.relu(self.v_conv(v))
            gv = F.relu(self.v_fc(pooled(v)))
            value = self.value_head(gv).squeeze(1)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
