import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 40
HEAD = 32
HEAD_K = 3
CRIT = 24
ATTN = False
LAYOUTS = [("tyx", 0, 0), ("txy", 0, 0), ("tyx", 1, 0), ("tyx", 0, 1)]
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}


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
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0
        self.k = self.n_place // yx if self.grid_ok else 0
        self.remap = nn.Linear(yx, yx)
        if self.grid_ok:
            n_mask = len(LAYOUTS) * self.k
        else:
            self.mask_embed = nn.Linear(self.n_actions, 4 * yx)
            n_mask = 4
        cin = 2 * c0 + n_mask + 2
        self.inp = nn.Conv2d(cin, STEM, 1)
        self.inorm = nn.GroupNorm(1, STEM)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        if ATTN:
            self.anorm = nn.LayerNorm(WIDTH)
            self.attn = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.hk = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid_ok:
            self.out = nn.Conv2d(HEAD, len(LAYOUTS) * self.k, 1)
        else:
            self.fb_proj = nn.Conv2d(HEAD, 4, 1)
            self.fb_out = nn.Linear(4 * yx, self.n_place)
        self.scale = (4.0 / len(LAYOUTS)) ** 0.5
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.extra = nn.Linear(64, max(self.n_extra, 1))
        self.c_inp = nn.Conv2d(cin, CRIT, 1)
        self.c_norm = nn.GroupNorm(1, CRIT)
        self.c_conv = nn.Conv2d(CRIT, CRIT, 3, padding=1)
        self.c_fc = nn.Sequential(nn.Linear(2 * CRIT, 64), nn.ReLU(), nn.Linear(64, 1))
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def forward(self, obs, mask):
        b = obs.shape[0]
        m = mask.float()
        feats = [obs, self.remap(obs.flatten(2)).reshape(b, -1, self.Y, self.X)]
        if self.grid_ok:
            mp = m[:, :self.n_place]
            for lay in LAYOUTS:
                feats.append(to_grid(mp, lay, self.k, self.Y, self.X))
        else:
            feats.append(self.mask_embed(m).reshape(b, 4, self.Y, self.X))
        feats.append(self.coords.expand(b, -1, -1, -1))
        z = torch.cat(feats, 1)
        x = F.relu(self.inorm(self.inp(z)))
        x = F.relu(self.stem2(x))
        x = F.relu(x + self.r1(x))
        if ATTN:
            t = x.flatten(2).transpose(1, 2)
            tn = self.anorm(t)
            a, _ = self.attn(tn, tn, tn, need_weights=False)
            x = x + a.transpose(1, 2).reshape(b, WIDTH, self.Y, self.X)
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hk(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        if self.grid_ok:
            maps = self.out(h)
            place = None
            for j, lay in enumerate(LAYOUTS):
                p = from_grid(maps[:, j * self.k:(j + 1) * self.k], lay)
                place = p if place is None else place + p
            place = place * self.scale
        else:
            place = self.fb_out(self.fb_proj(h).flatten(1))
        ex = self.extra(g)[:, :self.n_extra]
        logits = torch.cat([place, ex], 1)
        c = F.relu(self.c_norm(self.c_inp(z)))
        c = F.relu(c + self.c_conv(c))
        v = self.c_fc(torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1)).squeeze(1)
        return logits, v


def build_policy(spec):
    return Net(spec)
