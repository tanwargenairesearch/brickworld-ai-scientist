import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

OBS_VIEWS = "sym8"
ORDERS = ["tyx"]
USE_LN = False
CIRC = True
STEM = 64
WIDTH = 40
HEAD = 32
GLOB = 64
AXES = ["cyx", "cxy", "ycx", "xcy", "yxc", "xyc"]


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
        cells = Y * X
        sq = Y == X
        size = {"c": self.C, "y": Y, "x": X}
        self.reads = []
        self.syms = []
        if OBS_VIEWS == "axes6":
            for p in AXES:
                shp = [size[a] for a in p]
                perm = [0] + [1 + p.index(a) for a in "cyx"]
                self.reads.append((shp, perm))
            nv = len(self.reads)
        elif OBS_VIEWS == "sym8":
            self.syms = [(t, a, b) for t in (0, 1) for a in (0, 1) for b in (0, 1) if sq or not t]
            nv = len(self.syms)
        else:
            self.syms = [(0, 0, 0)]
            nv = 1
        obs_c = self.C * nv + 2
        self.grid = (self.n_place % cells) == 0
        self.K = self.n_place // cells if self.grid else int(spec.n_types)
        nE = len(ORDERS)
        if self.grid:
            mask_c = self.K * nE
        else:
            mask_c = 8
            self.mask_embed = nn.Linear(self.n_place, 8 * cells)
            self.out_lin = nn.Linear(self.K * nE * cells, self.n_place)
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
        pm = "circular" if CIRC else "zeros"
        mn = min(Y, X)
        d2 = max(1, min(2, mn // 2)) if CIRC else 1
        d3 = max(1, min(4, mn // 2)) if CIRC else 1
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode=pm)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=d2, dilation=d2, padding_mode=pm)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=d3, dilation=d3, padding_mode=pm)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, self.K * nE, 1)
        self.glob = nn.Linear(2 * WIDTH, GLOB)
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        if USE_LN:
            for lin in (self.hout, self.done_head, self.value_head):
                nn.init.orthogonal_(lin.weight, 0.01)
                nn.init.zeros_(lin.bias)
        ys = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def obs_views(self, obs):
        B = obs.shape[0]
        if OBS_VIEWS == "axes6":
            flat = obs.reshape(B, -1)
            v = [flat.reshape([B] + shp).permute(perm) for (shp, perm) in self.reads]
        else:
            v = [sym(obs, t, a, b) for (t, a, b) in self.syms]
        v.append(self.coords.expand(B, -1, -1, -1))
        return torch.cat(v, 1)

    def read(self, mp, od):
        B = mp.shape[0]
        if od == "yxt":
            return mp.reshape(B, self.Y, self.X, self.K).permute(0, 3, 1, 2)
        return mp.reshape(B, self.K, self.Y, self.X)

    def write(self, out, od):
        B = out.shape[0]
        if od == "yxt":
            return out.permute(0, 2, 3, 1).reshape(B, -1)
        return out.reshape(B, -1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        o = self.obs_views(obs)
        mp = mask[:, :self.n_place].float()
        if self.grid:
            m = torch.cat([self.read(mp, od) for od in ORDERS], 1)
        else:
            m = self.mask_embed(mp).reshape(B, 8, self.Y, self.X)
        if USE_LN:
            s = F.relu(self.obs_ln(self.obs_stem(o))) + F.relu(self.mask_ln(self.mask_stem(m)))
            x = F.relu(self.ln0(self.stem2(s)))
            x = F.relu(x + self.ln1(self.c1(x)))
            f = F.relu(x + self.ln2(self.c2(x)))
        else:
            s = F.relu(self.stem(torch.cat([o, m], 1)))
            x = F.relu(self.stem2(s))
            x = F.relu(x + self.c1(x))
            f = F.relu(x + self.c2(x))
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        K = self.K
        if self.grid:
            place = None
            for i, od in enumerate(ORDERS):
                pi = self.write(out[:, i * K:(i + 1) * K], od)
                place = pi if place is None else place + pi
        else:
            place = self.out_lin(out.reshape(B, -1))
        g = F.relu(self.glob(pooled(f)))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
