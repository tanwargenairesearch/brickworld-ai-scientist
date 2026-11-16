import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

# conv-head action layouts: (index order, DONE first). Orders: "tyx" type-major,
# "yxt" type-minor, "ytx" type-middle. HEAD: "conv", "dense" or "pointer".
COMPONENTS = [("ytx", False)]
HEAD = "conv"
OBS_VIEWS = "sym8"
USE_LN = False
GCTX = False
STEM = 64
WIDTH = 40
HC = 32
GLOB = 64
DC = 16
ME = 8
QSCALE = 10.0


def sym(m, t, fy, fx):
    if t:
        m = m.transpose(-1, -2)
    if fy:
        m = m.flip(-2)
    if fx:
        m = m.flip(-1)
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
        Y = int(spec.Y)
        X = int(spec.X)
        self.Y = Y
        self.X = X
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_act = int(spec.n_actions)
        cells = Y * X
        self.cells = cells
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        sq = Y == X
        if OBS_VIEWS == "sym8":
            self.syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or not t]
            n_views = len(self.syms)
        else:
            self.syms = []
            n_views = 3
        obs_c = self.C * n_views + 2
        self.mode = HEAD
        self.entries = []
        self.K = 0
        if HEAD == "conv":
            if self.n_place % cells == 0:
                self.mode = "grid"
                self.K = self.n_place // cells
                self.entries = [(od, df, None) for (od, df) in COMPONENTS]
            else:
                cands = [[(Y - dy + 1, X - dx + 1) for (dy, dx) in fps],
                         [(Y - dx + 1, X - dy + 1) for (dy, dx) in fps]]
                good = []
                for s in cands:
                    ok = all(h > 0 and w > 0 for (h, w) in s) and sum(h * w for (h, w) in s) == self.n_place
                    if ok and s not in good:
                        good.append(s)
                if len(good) > 0:
                    self.mode = "compact"
                    self.K = len(fps)
                    dfs = []
                    for (od, df) in COMPONENTS:
                        if df not in dfs:
                            dfs.append(df)
                    self.entries = [("tyx", df, s) for s in good for df in dfs]
                else:
                    self.mode = "dense"
        nE = len(self.entries)
        mask_c = self.K * nE if nE > 0 else ME
        if self.mode == "dense":
            self.mask_embed = nn.Linear(self.n_act, ME * cells)
        if self.mode == "pointer":
            self.q_in = nn.Parameter(torch.zeros(self.n_act, cells))
            self.u_in = nn.Parameter(torch.randn(ME, self.n_act))
            self.b_in = nn.Parameter(torch.zeros(ME * cells))
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
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        if GCTX:
            self.g1 = nn.Linear(2 * WIDTH, WIDTH)
            self.g2 = nn.Linear(2 * WIDTH, WIDTH)
        self.glob = nn.Linear(2 * WIDTH, GLOB)
        self.value_head = nn.Linear(GLOB, 1)
        if self.mode in ("grid", "compact"):
            self.h3 = nn.Conv2d(WIDTH, HC, 3, padding=1)
            self.h1 = nn.Conv2d(WIDTH, HC, 1)
            self.hout = nn.Conv2d(HC, self.K * nE, 1)
            self.done_heads = nn.ModuleList([nn.Linear(GLOB, 1) for _ in range(nE)])
            outs = [self.hout] + list(self.done_heads)
        else:
            self.hp = nn.Conv2d(WIDTH, DC, 1)
            if self.mode == "dense":
                self.out = nn.Linear(DC * cells + GLOB, self.n_act)
                outs = [self.out]
            else:
                self.q_out = nn.Parameter(torch.zeros(self.n_act, cells))
                self.pw = nn.Linear(DC, self.n_act)
                self.pg = nn.Linear(GLOB, self.n_act)
                outs = [self.pw, self.pg]
        if USE_LN:
            for lin in outs + [self.value_head]:
                nn.init.orthogonal_(lin.weight, 0.01)
                nn.init.zeros_(lin.bias)
        ys = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def obs_views(self, obs):
        B = obs.shape[0]
        if OBS_VIEWS == "sym8":
            v = [sym(obs, t, a, b) for (t, a, b) in self.syms]
        else:
            v = [obs,
                 obs.reshape(B, self.Y, self.X, self.C).permute(0, 3, 1, 2),
                 obs.permute(0, 2, 3, 1).reshape(B, self.C, self.Y, self.X)]
        v.append(self.coords.expand(B, -1, -1, -1))
        return torch.cat(v, 1)

    def read(self, mp, od, s):
        B = mp.shape[0]
        K, Y, X = self.K, self.Y, self.X
        if s is not None:
            parts = []
            i = 0
            for (h, w) in s:
                c = mp[:, i:i + h * w].reshape(B, 1, h, w)
                parts.append(F.pad(c, (0, X - w, 0, Y - h)))
                i += h * w
            return torch.cat(parts, 1)
        if od == "yxt":
            return mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
        if od == "ytx":
            return mp.reshape(B, Y, K, X).permute(0, 2, 1, 3)
        return mp.reshape(B, K, Y, X)

    def write(self, out, od, s):
        B = out.shape[0]
        if s is not None:
            parts = []
            for t, (h, w) in enumerate(s):
                parts.append(out[:, t, :h, :w].reshape(B, -1))
            return torch.cat(parts, 1)
        if od == "yxt":
            return out.permute(0, 2, 3, 1).reshape(B, -1)
        if od == "ytx":
            return out.permute(0, 2, 1, 3).reshape(B, -1)
        return out.reshape(B, -1)

    def block(self, x, conv, gl, ln):
        y = conv(x)
        if gl is not None:
            y = y + gl(pooled(x))[:, :, None, None]
        if ln is not None:
            y = ln(y)
        return F.relu(x + y)

    def forward(self, obs, mask):
        B = obs.shape[0]
        ov = self.obs_views(obs)
        mf = mask.float()
        if self.mode in ("grid", "compact"):
            parts = []
            for (od, df, s) in self.entries:
                mp = mf[:, 1:self.n_place + 1] if df else mf[:, :self.n_place]
                parts.append(self.read(mp, od, s))
            m = torch.cat(parts, 1)
        elif self.mode == "dense":
            m = self.mask_embed(mf).reshape(B, ME, self.Y, self.X)
        else:
            S = torch.softmax(self.q_in * QSCALE, dim=1)
            W = torch.einsum("ea,ac->eca", self.u_in, S).reshape(ME * self.cells, self.n_act)
            m = F.linear(mf, W, self.b_in).reshape(B, ME, self.Y, self.X)
        if USE_LN:
            s0 = F.relu(self.obs_ln(self.obs_stem(ov))) + F.relu(self.mask_ln(self.mask_stem(m)))
            x = F.relu(self.ln0(self.stem2(s0)))
            l1, l2 = self.ln1, self.ln2
        else:
            s0 = F.relu(self.stem(torch.cat([ov, m], 1)))
            x = F.relu(self.stem2(s0))
            l1, l2 = None, None
        g1 = self.g1 if GCTX else None
        g2 = self.g2 if GCTX else None
        x = self.block(x, self.c1, g1, l1)
        f = self.block(x, self.c2, g2, l2)
        g = F.relu(self.glob(pooled(f)))
        if self.mode in ("grid", "compact"):
            h = F.relu(self.h3(f) + self.h1(f))
            out = self.hout(h)
            K = self.K
            logits = None
            for i, (od, df, s) in enumerate(self.entries):
                p = self.write(out[:, i * K:(i + 1) * K], od, s)
                d = self.done_heads[i](g)
                li = torch.cat([d, p], 1) if df else torch.cat([p, d], 1)
                logits = li if logits is None else logits + li
        elif self.mode == "dense":
            h = F.relu(self.hp(f)).flatten(1)
            logits = self.out(torch.cat([h, g], 1))
        else:
            h = F.relu(self.hp(f)).flatten(1)
            S = torch.softmax(self.q_out * QSCALE, dim=1)
            W = torch.einsum("ac,ad->adc", S, self.pw.weight).reshape(self.n_act, -1)
            logits = F.linear(h, W, self.pw.bias) + self.pg(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
