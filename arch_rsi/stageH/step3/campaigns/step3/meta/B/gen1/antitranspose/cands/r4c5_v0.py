import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [("TYX", 0, 0), ("TYX", 0, 1), ("TYX", 1, 0), ("TYX", 1, 1), ("TXY", 0, 0), ("TXY", 0, 1), ("TXY", 1, 0), ("TXY", 1, 1), ("YXT", 0, 0), ("YXT", 0, 1), ("YXT", 1, 0), ("YXT", 1, 1), ("XYT", 0, 0), ("XYT", 0, 1), ("XYT", 1, 0), ("XYT", 1, 1)]
POOLHEAD = False
NBLOCKS = 2
OUT_GAIN = 0.01
STEM = 64
WIDTH = 40
HEADC = 32
GLOB = 64
MASK_EMB = 16
PROJ = 32
HID = 32


def read_grid(v, order, names, sizes, fy, fx):
    b = v.shape[0]
    g = v.reshape([b] + [sizes[a] for a in order])
    g = g.permute([0] + [1 + order.index(a) for a in names])
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def write_grid(g, order, names, fy, fx):
    if fx:
        g = g.flip(3)
    if fy:
        g = g.flip(2)
    g = g.permute([0] + [1 + names.index(a) for a in order])
    return g.reshape(g.shape[0], -1)


class ChanNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ny = int(spec.Y)
        self.nx = int(spec.X)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.cin_obs = int(spec.in_channels)
        self.grid_ok = self.n_place == self.nt * self.ny * self.nx
        cobs = self.cin_obs + 2
        if self.grid_ok:
            n_mask = len(LAYOUTS) * self.nt
        else:
            n_mask = MASK_EMB
            self.mask_emb = nn.Linear(self.n_actions, MASK_EMB * self.ny * self.nx)
        self.obs_stem = nn.Conv2d(cobs, STEM, 1)
        self.mask_stem = nn.Conv2d(n_mask, STEM, 1)
        self.no = ChanNorm(STEM)
        self.nm = ChanNorm(STEM)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.n1 = ChanNorm(WIDTH)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLOCKS)])
        self.rn = nn.ModuleList([ChanNorm(WIDTH) for _ in range(NBLOCKS)])
        self.hk = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        self.hs = nn.Conv2d(WIDTH, HEADC, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)
        out_layers = [self.done_head]
        self.use_pool = bool(POOLHEAD) and self.grid_ok and len(spec.footprints) == self.nt
        if self.grid_ok:
            self.outs = nn.Conv2d(HEADC, len(LAYOUTS) * self.nt, 1)
            out_layers.append(self.outs)
            if self.use_pool:
                self.pool = PoolHead(WIDTH, spec.footprints, PROJ, HID)
                out_layers.extend(list(self.pool.outs))
        else:
            self.fc_place = nn.Linear(HEADC * self.ny * self.nx, self.n_place)
            out_layers.append(self.fc_place)
        for layer in out_layers:
            nn.init.orthogonal_(layer.weight, gain=OUT_GAIN)
            nn.init.zeros_(layer.bias)

    def forward(self, obs, mask):
        b = obs.shape[0]
        ny, nx, nt, npl = self.ny, self.nx, self.nt, self.n_place
        cy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(b, 1, ny, nx)
        cx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(b, 1, ny, nx)
        zo = torch.cat([obs, cy, cx], 1)
        m = mask.to(obs.dtype)
        if self.grid_ok:
            sm = {"T": nt, "Y": ny, "X": nx}
            mp = m[:, :npl]
            zm = torch.cat([read_grid(mp, o, "TYX", sm, fy, fx) for o, fy, fx in LAYOUTS], 1)
        else:
            zm = self.mask_emb(m).view(b, MASK_EMB, ny, nx)
        s = F.relu(self.no(self.obs_stem(zo)) + self.nm(self.mask_stem(zm)))
        f = F.relu(self.n1(self.stem2(s)))
        for conv, nrm in zip(self.res, self.rn):
            f = F.relu(f + nrm(conv(f)))
        h = F.relu(self.hk(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        dn = self.done_head(g)
        if self.grid_ok:
            maps = self.outs(h)
            pl = None
            for i, (o, fy, fx) in enumerate(LAYOUTS):
                part = write_grid(maps[:, i * nt:(i + 1) * nt], o, "TYX", fy, fx)
                pl = part if pl is None else pl + part
            if self.use_pool:
                pl = pl + self.pool(f)
        else:
            pl = self.fc_place(h.flatten(1))
        logits = torch.cat([pl, dn], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
