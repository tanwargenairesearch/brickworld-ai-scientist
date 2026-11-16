import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 4,
          "ppo_epochs": 1, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
DILS = [1, 1]
HEAD_C = 32
HEAD_K = 3
GHID = 64
MASK_HEAD = False
LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
OBS_SYMS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]


def flip_dims(fy, fx):
    d = []
    if fy:
        d.append(2)
    if fx:
        d.append(3)
    return d


def to_grid(m, T, Y, X, lay):
    order, trans, fy, fx = lay
    B = m.shape[0]
    if order == 0:
        if trans == 0:
            g = m.reshape(B, T, Y, X)
        else:
            g = m.reshape(B, T, X, Y).transpose(2, 3)
    else:
        if trans == 0:
            g = m.reshape(B, Y, X, T).permute(0, 3, 1, 2)
        else:
            g = m.reshape(B, X, Y, T).permute(0, 3, 2, 1)
    d = flip_dims(fy, fx)
    if d:
        g = torch.flip(g, d)
    return g


def from_grid(g, lay):
    order, trans, fy, fx = lay
    B = g.shape[0]
    d = flip_dims(fy, fx)
    if d:
        g = torch.flip(g, d)
    if order == 0:
        if trans == 0:
            out = g
        else:
            out = g.transpose(2, 3)
    else:
        if trans == 0:
            out = g.permute(0, 2, 3, 1)
        else:
            out = g.permute(0, 3, 2, 1)
    return out.reshape(B, -1)


def sym_obs(o, s):
    trans, fy, fx = s
    if trans:
        o = o.transpose(2, 3)
    d = flip_dims(fy, fx)
    if d:
        o = torch.flip(o, d)
    return o


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.n_extra = self.n_actions - self.n_place
        self.grid_ok = self.n_place == self.T * self.Y * self.X
        self.obs_syms = [s for s in OBS_SYMS if s[0] == 0 or self.Y == self.X]
        self.layouts = list(LAYOUTS)
        YX = self.Y * self.X
        if self.grid_ok:
            mask_ch = len(self.layouts) * self.T
        else:
            mask_ch = 8
            self.mask_embed = nn.Linear(self.n_actions, 8 * YX)
        cin = len(self.obs_syms) * spec.in_channels + mask_ch + 2
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILS])
        self.hconv = nn.Conv2d(WIDTH, HEAD_C, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEAD_C, 1)
        self.use_mh = MASK_HEAD
        if self.use_mh:
            self.hmask = nn.Conv2d(mask_ch, HEAD_C, 1)
        if self.grid_ok:
            self.hout = nn.Conv2d(HEAD_C, len(self.layouts) * self.T, 1)
        else:
            self.hred = nn.Conv2d(HEAD_C, 4, 1)
            self.hlin = nn.Linear(4 * YX, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GHID), nn.ReLU())
        self.done_head = nn.Linear(GHID, self.n_extra)
        self.value_head = nn.Linear(GHID, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        npl = self.n_place
        mf = mask.float()
        parts = [sym_obs(obs, s) for s in self.obs_syms]
        mparts = []
        if self.grid_ok:
            mp = mf[:, :npl]
            for lay in self.layouts:
                mparts.append(to_grid(mp, T, Y, X, lay))
        else:
            mparts.append(self.mask_embed(mf).view(B, 8, Y, X))
        mg = torch.cat(mparts, 1)
        parts.append(mg)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(yy)
        parts.append(xx)
        x = torch.cat(parts, 1)
        f = F.relu(self.stem2(F.relu(self.stem1(x))))
        for conv in self.res:
            f = F.relu(f + conv(f))
        hp = self.hconv(f) + self.hskip(f)
        if self.use_mh:
            hp = hp + self.hmask(mg)
        h = F.relu(hp)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        if self.grid_ok:
            maps = self.hout(h)
            p = 0
            for k, lay in enumerate(self.layouts):
                p = p + from_grid(maps[:, k * T:(k + 1) * T], lay)
        else:
            p = self.hlin(F.relu(self.hred(h)).flatten(1))
        logits = torch.cat([p, d], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
