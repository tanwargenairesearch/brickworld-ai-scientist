import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 40
HEAD = 32
HEAD_K = 3
LAYOUTS = [("tyx", 0, 0), ("txy", 0, 0), ("tyx", 1, 0), ("tyx", 0, 1)]
REMAP = True
CH_REMAP = False
SYM_OBS = False
MASK_REMAP = False
OUT_REMAP = False
FLAT = 0
GLOBAL = True
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}


def to_grid(flat, lay, k, ny, nx):
    order, fy, fx = lay
    dims = {"t": k, "y": ny, "x": nx}
    b = flat.shape[0]
    g = flat.reshape(b, dims[order[0]], dims[order[1]], dims[order[2]])
    g = g.permute(0, order.index("t") + 1, order.index("y") + 1, order.index("x") + 1)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    order, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    g = g.permute(0, "tyx".index(order[0]) + 1, "tyx".index(order[1]) + 1, "tyx".index(order[2]) + 1)
    return g.reshape(g.shape[0], -1)


def sym_views(obs, square):
    views = [obs.flip(2), obs.flip(3), obs.flip(2).flip(3)]
    if square:
        t = obs.transpose(2, 3)
        views = views + [t, t.flip(2), t.flip(3), t.flip(2).flip(3)]
    return views


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.n_extra = self.n_actions - self.n_place
        yx = self.Y * self.X
        self.yx = yx
        c0 = int(spec.in_channels)
        self.square = self.Y == self.X
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0
        self.k = self.n_place // yx if self.grid_ok else 0
        n_obs = c0
        if REMAP:
            self.remap = nn.Linear(yx, yx)
            n_obs += c0
        if CH_REMAP:
            self.ch_remap = nn.Conv1d(c0 * yx, c0 * yx, 1, groups=c0)
            n_obs += c0
        if SYM_OBS:
            n_obs += c0 * (7 if self.square else 3)
        if self.grid_ok:
            n_mask = len(LAYOUTS) * self.k
            if MASK_REMAP:
                self.mask_remap = nn.Linear(yx, yx)
                n_mask += self.k
        else:
            self.mask_embed = nn.Linear(self.n_actions, 4 * yx)
            n_mask = 4
        cin = n_obs + n_mask + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        if GLOBAL:
            self.d1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
            self.gb = nn.Linear(2 * WIDTH, WIDTH)
        self.hk = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEAD, 1)
        n_terms = len(LAYOUTS)
        if self.grid_ok:
            n_maps = len(LAYOUTS) * self.k
            if OUT_REMAP:
                n_maps += self.k
                n_terms += 1
                self.out_remap = nn.Linear(yx, yx)
            self.out = nn.Conv2d(HEAD, n_maps, 1)
        else:
            self.fb_proj = nn.Conv2d(HEAD, 4, 1)
            self.fb_out = nn.Linear(4 * yx, self.n_place)
        if FLAT > 0:
            self.flat_proj = nn.Conv2d(HEAD, FLAT, 1)
            self.flat_out = nn.Linear(FLAT * yx, self.n_place)
        self.scale = (4.0 / n_terms) ** 0.5
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.extra = nn.Linear(64, max(self.n_extra, 1))
        self.value_head = nn.Linear(64, 1)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def forward(self, obs, mask):
        b = obs.shape[0]
        m = mask.float()
        feats = [obs]
        if REMAP:
            feats.append(self.remap(obs.flatten(2)).reshape(b, -1, self.Y, self.X))
        if CH_REMAP:
            feats.append(self.ch_remap(obs.reshape(b, -1, 1)).reshape(b, -1, self.Y, self.X))
        if SYM_OBS:
            feats = feats + sym_views(obs, self.square)
        if self.grid_ok:
            mp = m[:, :self.n_place]
            for lay in LAYOUTS:
                feats.append(to_grid(mp, lay, self.k, self.Y, self.X))
            if MASK_REMAP:
                feats.append(self.mask_remap(mp.reshape(b, self.k, self.yx)).reshape(b, self.k, self.Y, self.X))
        else:
            feats.append(self.mask_embed(m).reshape(b, 4, self.Y, self.X))
        feats.append(self.coords.expand(b, -1, -1, -1))
        x = self.stem(torch.cat(feats, 1))
        x = F.relu(x + self.r1(x))
        if GLOBAL:
            gl = self.gb(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
            x = F.relu(x + self.d1(x) + gl[:, :, None, None])
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hk(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        if self.grid_ok:
            maps = self.out(h)
            place = None
            for j, lay in enumerate(LAYOUTS):
                p = from_grid(maps[:, j * self.k:(j + 1) * self.k], lay)
                place = p if place is None else place + p
            if OUT_REMAP:
                q = maps[:, len(LAYOUTS) * self.k:].reshape(b, self.k, self.yx)
                place = place + self.out_remap(q).reshape(b, -1)
            place = place * self.scale
        else:
            place = self.fb_out(self.fb_proj(h).flatten(1))
        if FLAT > 0:
            place = place + self.flat_out(self.flat_proj(h).flatten(1))
        ex = self.extra(g)[:, :self.n_extra]
        logits = torch.cat([place, ex], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
