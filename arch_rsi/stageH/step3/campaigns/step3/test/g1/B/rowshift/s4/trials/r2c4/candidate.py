import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 40
HEAD = 32
HEAD_K = 3
LAYOUTS = [("tyx", 0, 0), ("txy", 0, 0), ("ytx", 0, 0), ("xty", 0, 0), ("yxt", 0, 0), ("xyt", 0, 0), ("tyx", 1, 0), ("tyx", 0, 1)]
OFFSETS = [0, 1]
OBS_VIEWS = ["yxc", "xyc", "ycx"]
REMAP = True
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


def obs_view(obs, v, ny, nx):
    b, c = obs.shape[0], obs.shape[1]
    dims = {"c": c, "y": ny, "x": nx}
    g = obs.reshape(b, dims[v[0]], dims[v[1]], dims[v[2]])
    return g.permute(0, v.index("c") + 1, v.index("y") + 1, v.index("x") + 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.n_extra = self.n_actions - self.n_place
        yx = self.Y * self.X
        c0 = int(spec.in_channels)
        self.grid_ok = self.n_place > 0 and self.n_place % yx == 0
        self.k = self.n_place // yx if self.grid_ok else 0
        offs = [o for o in OFFSETS if o + self.n_place <= self.n_actions]
        self.offsets = offs if len(offs) > 0 else [0]
        if self.grid_ok:
            n_mask = len(LAYOUTS) * len(self.offsets) * self.k
        else:
            self.offsets = [0]
            self.mask_embed = nn.Linear(self.n_actions, 4 * yx)
            n_mask = 4
        n_obs = c0 * (1 + len(OBS_VIEWS) + (1 if REMAP else 0))
        if REMAP:
            self.remap = nn.Linear(yx, yx)
        cin = n_obs + n_mask + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, HEAD, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEAD, 1)
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(HEAD, len(LAYOUTS) * self.k, 1) for _ in self.offsets])
        else:
            self.fb_proj = nn.Conv2d(HEAD, 4, 1)
            self.fb_out = nn.Linear(4 * yx, self.n_place)
        self.scale = (4.0 / len(LAYOUTS)) ** 0.5
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.extra = nn.ModuleList([nn.Linear(64, max(self.n_extra, 1)) for _ in self.offsets])
        self.value_head = nn.Linear(64, 1)
        ys = torch.linspace(-1.0, 1.0, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).contiguous())

    def forward(self, obs, mask):
        b = obs.shape[0]
        m = mask.float()
        feats = [obs]
        for v in OBS_VIEWS:
            feats.append(obs_view(obs, v, self.Y, self.X))
        if REMAP:
            feats.append(self.remap(obs.flatten(2)).reshape(b, -1, self.Y, self.X))
        if self.grid_ok:
            for o in self.offsets:
                mp = m[:, o:o + self.n_place]
                for lay in LAYOUTS:
                    feats.append(to_grid(mp, lay, self.k, self.Y, self.X))
        else:
            feats.append(self.mask_embed(m).reshape(b, 4, self.Y, self.X))
        feats.append(self.coords.expand(b, -1, -1, -1))
        x = self.stem(torch.cat(feats, 1))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hk(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        logits = None
        for i, o in enumerate(self.offsets):
            if self.grid_ok:
                maps = self.outs[i](h)
                place = None
                for j, lay in enumerate(LAYOUTS):
                    p = from_grid(maps[:, j * self.k:(j + 1) * self.k], lay)
                    place = p if place is None else place + p
                place = place * self.scale
            else:
                place = self.fb_out(self.fb_proj(h).flatten(1))
            ex = self.extra[i](g)[:, :self.n_extra]
            vec = torch.cat([ex[:, :o], place, ex[:, o:]], 1)
            logits = vec if logits is None else logits + vec
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
