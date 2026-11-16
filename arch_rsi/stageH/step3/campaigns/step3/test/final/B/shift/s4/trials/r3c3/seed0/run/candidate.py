import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
OBS_VIEWS = ["orig"]
OBS_SYM = False
USE_LN = True
OUT_GAIN = 0.01
TYPE_MINOR = False
DENSE_OBS = True
SEP_CRITIC = False
GACT_GAIN = 0.01

STEM = 64
WIDTH = 40
HEADC = 32
GH = 64
DCH_MAX = 8
DENSE_PARAMS = 800000
CRIT = 32


def fwd_t(v, t, fy, fx):
    if t:
        v = v.transpose(-1, -2)
    if fy:
        v = v.flip(-2)
    if fx:
        v = v.flip(-1)
    return v


def inv_t(v, t, fy, fx):
    if fx:
        v = v.flip(-1)
    if fy:
        v = v.flip(-2)
    if t:
        v = v.transpose(-1, -2)
    return v


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def norm(c):
    return ChanLN(c) if USE_LN else nn.Identity()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        Y, X = self.Y, self.X
        cells = Y * X
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        compact = sum(max(Y - dy + 1, 0) * max(X - dx + 1, 0) for dy, dx in self.fps)
        if self.n_place % cells != 0 and self.n_place == compact and not TYPE_MINOR:
            self.compact = True
            self.K = len(self.fps)
        else:
            self.compact = False
            self.K = (self.n_place + cells - 1) // cells
        sq = Y == X
        self.layouts = [l for l in LAYOUTS if sq or l[0] == 0]
        if OBS_SYM:
            self.syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or t == 0]
        else:
            self.syms = [(0, 0, 0)]
        c_obs = self.C * len(OBS_VIEWS) * len(self.syms) + 2
        c_mask = self.K * len(self.layouts)
        self.dch = 0
        if DENSE_OBS:
            self.dch = max(1, min(DCH_MAX, DENSE_PARAMS // (self.C * cells * cells)))
            self.dense = nn.Linear(self.C * cells, self.dch * cells)
            self.dense_ln = ChanLN(self.dch)
        if USE_LN:
            self.obs_stem = nn.Sequential(nn.Conv2d(c_obs, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.mask_stem = nn.Sequential(nn.Conv2d(c_mask, STEM, 1), ChanLN(STEM), nn.ReLU())
            self.stem2 = nn.Sequential(nn.Conv2d(2 * STEM + self.dch, WIDTH, 1), ChanLN(WIDTH), nn.ReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(c_obs + c_mask + self.dch, STEM, 1), nn.ReLU(),
                                      nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n1 = norm(WIDTH)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = norm(WIDTH)
        self.h3 = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEADC, 1)
        self.hout = nn.Conv2d(HEADC, self.K * len(self.layouts), 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        self.gact = nn.Linear(GH, self.n_actions)
        nn.init.orthogonal_(self.gact.weight, gain=GACT_GAIN)
        nn.init.zeros_(self.gact.bias)
        if SEP_CRITIC:
            self.crit = nn.Sequential(nn.Conv2d(c_obs, CRIT, 1), nn.ReLU(),
                                      nn.Conv2d(CRIT, CRIT, 3, padding=1), nn.ReLU())
            self.crit_fc = nn.Sequential(nn.Linear(2 * CRIT, GH), nn.ReLU(), nn.Linear(GH, 1))
        if OUT_GAIN > 0:
            for m in (self.hout, self.done_head, self.value_head):
                nn.init.orthogonal_(m.weight, gain=OUT_GAIN)
                nn.init.zeros_(m.bias)

    def to_maps(self, m, B):
        Y, X, K = self.Y, self.X, self.K
        if self.compact:
            maps = []
            off = 0
            for dy, dx in self.fps:
                h = Y - dy + 1
                w = X - dx + 1
                seg = m[:, off:off + h * w].reshape(B, 1, h, w)
                maps.append(F.pad(seg, (0, X - w, 0, Y - h)))
                off += h * w
            return torch.cat(maps, dim=1)
        cells = Y * X
        if K * cells > self.n_place:
            m = F.pad(m, (0, K * cells - self.n_place))
        if TYPE_MINOR:
            return m.reshape(B, Y, X, K).permute(0, 3, 1, 2)
        return m.reshape(B, K, Y, X)

    def to_flat(self, p, B):
        Y, X = self.Y, self.X
        if self.compact:
            segs = []
            for t, (dy, dx) in enumerate(self.fps):
                h = Y - dy + 1
                w = X - dx + 1
                segs.append(p[:, t, :h, :w].reshape(B, h * w))
            return torch.cat(segs, dim=1)
        if TYPE_MINOR:
            p = p.permute(0, 2, 3, 1).reshape(B, -1)
        else:
            p = p.reshape(B, -1)
        return p[:, :self.n_place]

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        views = []
        for v in OBS_VIEWS:
            if v == "chlast":
                views.append(obs.reshape(B, Y, X, self.C).permute(0, 3, 1, 2))
            else:
                views.append(obs)
        ob = torch.cat([fwd_t(v, *s) for v in views for s in self.syms] + [ys, xs], dim=1)
        m = mask[:, :self.n_place].to(obs.dtype)
        mm = self.to_maps(m, B)
        mk = torch.cat([fwd_t(mm, *l) for l in self.layouts], dim=1)
        extra = []
        if DENSE_OBS:
            d = self.dense(obs.reshape(B, -1)).reshape(B, self.dch, Y, X)
            extra.append(F.relu(self.dense_ln(d)))
        if USE_LN:
            f = self.stem2(torch.cat([self.obs_stem(ob), self.mask_stem(mk)] + extra, dim=1))
        else:
            f = self.stem(torch.cat([ob, mk] + extra, dim=1))
        f = F.relu(self.n1(f + self.r1(f)))
        f = F.relu(self.n2(f + self.r2(f)))
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        place = None
        for i, l in enumerate(self.layouts):
            p = self.to_flat(inv_t(out[:, i * K:(i + 1) * K], *l), B)
            place = p if place is None else place + p
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.gact(g)
        if SEP_CRITIC:
            c = self.crit(ob)
            value = self.crit_fc(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1)).squeeze(1)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
