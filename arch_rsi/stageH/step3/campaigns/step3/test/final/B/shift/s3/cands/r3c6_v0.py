import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

# (order, transpose, yflip, xflip); order is the flat action index order
LAYOUTS = [("yxt", False, False, False), ("yxt", True, False, False), ("yxt", False, True, False), ("yxt", False, False, True)]
DONE_FIRST = False
OBS_SYM = False
OBS_CL = True
SEP_CRITIC = False

ALL8 = [(False, False, False), (False, False, True), (False, True, False), (False, True, True),
        (True, False, False), (True, False, True), (True, True, False), (True, True, True)]

STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
MEMB = 8
CW = 32


def apply_layout(t, tr, fy, fx):
    if tr:
        t = t.transpose(-1, -2)
    if fy:
        t = t.flip(-2)
    if fx:
        t = t.flip(-1)
    return t


def invert_layout(t, tr, fy, fx):
    if fx:
        t = t.flip(-1)
    if fy:
        t = t.flip(-2)
    if tr:
        t = t.transpose(-1, -2)
    return t


def to_grid(mp, order, B, K, Y, X):
    if order == "yxt":
        return mp.reshape(B, Y, X, K).permute(0, 3, 1, 2)
    if order == "ytx":
        return mp.reshape(B, Y, K, X).permute(0, 2, 1, 3)
    return mp.reshape(B, K, Y, X)


def from_grid(g, order, B):
    if order == "yxt":
        return g.permute(0, 2, 3, 1).reshape(B, -1)
    if order == "ytx":
        return g.permute(0, 2, 1, 3).reshape(B, -1)
    return g.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.Y, self.X = Y, X
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        K = self.n_place // (Y * X)
        self.K = K
        self.grid_ok = K >= 1 and K * Y * X == self.n_place
        sq = Y == X
        lays = [l for l in LAYOUTS if sq or not l[1]]
        if len(lays) == 0:
            lays = [("yxt", False, False, False)]
        self.layouts = lays
        if OBS_SYM:
            self.views = [l for l in ALL8 if sq or not l[0]]
        else:
            self.views = [(False, False, False)]
        nv = len(self.views) + (1 if OBS_CL else 0)
        c_obs = self.C * nv + 2
        if self.grid_ok:
            c_mask = K * len(self.layouts)
        else:
            c_mask = MEMB
            self.mask_lin = nn.Linear(self.n_place, MEMB * Y * X)
        self.stem = nn.Sequential(nn.Conv2d(c_obs + c_mask, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.h3 = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid_ok:
            self.hout = nn.Conv2d(HEAD, K * len(self.layouts), 1)
        else:
            self.hout = nn.Conv2d(HEAD, 4, 1)
            self.hlin = nn.Linear(4 * Y * X, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        if SEP_CRITIC:
            self.critic = nn.Sequential(nn.Conv2d(self.C + 2, CW, 1), nn.ReLU(), nn.Conv2d(CW, CW, 3, padding=1), nn.ReLU())
            self.cfc = nn.Sequential(nn.Linear(2 * CW, GH), nn.ReLU(), nn.Linear(GH, 1))
        else:
            self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        views = [apply_layout(obs, v[0], v[1], v[2]) for v in self.views]
        if OBS_CL:
            views.append(obs.reshape(B, Y, X, self.C).permute(0, 3, 1, 2))
        o = torch.cat(views + [ys, xs], dim=1)
        mf = mask.to(obs.dtype)
        if DONE_FIRST:
            mp = mf[:, 1:1 + self.n_place]
        else:
            mp = mf[:, :self.n_place]
        if self.grid_ok:
            m = torch.cat([apply_layout(to_grid(mp, l[0], B, K, Y, X), l[1], l[2], l[3]) for l in self.layouts], dim=1)
        else:
            m = self.mask_lin(mp).view(B, MEMB, Y, X)
        x = self.stem(torch.cat([o, m], dim=1))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.h3(f) + self.h1(f))
        out = self.hout(h)
        if self.grid_ok:
            place = None
            for i, l in enumerate(self.layouts):
                oi = from_grid(invert_layout(out[:, i * K:(i + 1) * K], l[1], l[2], l[3]), l[0], B)
                place = oi if place is None else place + oi
        else:
            place = self.hlin(out.flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g)
        if DONE_FIRST:
            logits = torch.cat([done, place], dim=1)
        else:
            logits = torch.cat([place, done], dim=1)
        if SEP_CRITIC:
            c = self.critic(torch.cat([obs, ys, xs], dim=1))
            v = self.cfc(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1)).squeeze(1)
        else:
            v = self.value_head(g).squeeze(1)
        return logits, v


def build_policy(spec):
    return Net(spec)
