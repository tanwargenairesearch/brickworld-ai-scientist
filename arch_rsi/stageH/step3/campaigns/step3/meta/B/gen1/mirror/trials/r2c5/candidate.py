import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 1, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

# action-layout hedges: (type_minor, transpose, flip_y, flip_x)
LAYOUTS = [(0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
# obs views: (transpose, flip_y, flip_x)
OBS_VIEWS = [(0, 0, 0), (1, 0, 0)]
HEAD_K = 3
HEAD_C = 32
STEM = 64
WIDTH = 40
N_RES = 2


def to_grid(m, k, Y, X, lay):
    mi, tr, fy, fx = lay
    B = m.shape[0]
    if mi == 0 and tr == 0:
        g = m.reshape(B, k, Y, X)
    elif mi == 0:
        g = m.reshape(B, k, X, Y).transpose(2, 3)
    elif tr == 0:
        g = m.reshape(B, Y, X, k).permute(0, 3, 1, 2)
    else:
        g = m.reshape(B, X, Y, k).permute(0, 3, 2, 1)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    mi, tr, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if mi == 0 and tr == 1:
        g = g.transpose(2, 3)
    elif mi == 1 and tr == 0:
        g = g.permute(0, 2, 3, 1)
    elif mi == 1:
        g = g.permute(0, 3, 2, 1)
    return g.reshape(g.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.cin = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        yx = self.Y * self.X
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0
        self.K = self.n_place // yx if self.grid_ok else 1
        sq = self.Y == self.X
        self.views = [v for v in OBS_VIEWS if v[0] == 0 or sq]
        nl = len(LAYOUTS)
        if self.grid_ok:
            mask_ch = nl * self.K
            self.mask_lin = None
        else:
            mask_ch = 4
            self.mask_lin = nn.Linear(self.n_actions, 4 * yx)
        self.s1 = nn.Conv2d(self.cin * len(self.views) + mask_ch + 2, STEM, 1)
        self.s2 = nn.Conv2d(STEM, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.hc = nn.Conv2d(WIDTH, HEAD_C, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEAD_C, 1)
        if self.grid_ok:
            self.out = nn.Conv2d(HEAD_C, nl * self.K, 1)
            self.out_lin = None
        else:
            self.out = nn.Conv2d(HEAD_C, 4, 1)
            self.out_lin = nn.Linear(4 * yx, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        m = mask.to(obs.dtype)
        parts = []
        for tr, fy, fx in self.views:
            o = obs
            if tr:
                o = o.transpose(2, 3)
            if fy:
                o = o.flip(2)
            if fx:
                o = o.flip(3)
            parts.append(o)
        if self.grid_ok:
            mp = m[:, :self.n_place]
            for lay in LAYOUTS:
                parts.append(to_grid(mp, self.K, Y, X, lay))
        else:
            parts.append(self.mask_lin(m).view(B, 4, Y, X))
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(ys)
        parts.append(xs)
        h = torch.cat(parts, 1)
        f = F.relu(self.s2(F.relu(self.s1(h))))
        for c in self.res:
            f = F.relu(f + c(f))
        hd = F.relu(self.hc(f) + self.hs(f))
        o = self.out(hd)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        if self.grid_ok:
            place = None
            for j, lay in enumerate(LAYOUTS):
                p = from_grid(o[:, j * self.K:(j + 1) * self.K], lay)
                place = p if place is None else place + p
        else:
            place = self.out_lin(o.flatten(1))
        full = torch.cat([place, d], 1)
        n = full.shape[1]
        if n < self.n_actions:
            full = F.pad(full, (0, self.n_actions - n))
        elif n > self.n_actions:
            full = full[:, :self.n_actions]
        return full, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
