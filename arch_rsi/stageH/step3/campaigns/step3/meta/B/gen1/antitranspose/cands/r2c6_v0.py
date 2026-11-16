import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

# obs views: (memory order of C, Y, X; flip_y; flip_x)
OBS_VIEWS = [("CYX", 0, 0), ("CYX", 0, 1), ("CYX", 1, 0), ("CYX", 1, 1), ("CXY", 0, 0), ("CXY", 0, 1), ("CXY", 1, 0), ("CXY", 1, 1)]
# mask-input / output-map layouts: (memory order of T, Y, X; flip_y; flip_x)
LAYOUTS = [("TYX", 0, 0), ("TYX", 0, 1), ("TYX", 1, 0), ("TYX", 1, 1), ("TXY", 0, 0), ("TXY", 0, 1), ("TXY", 1, 0), ("TXY", 1, 1), ("YXT", 0, 0), ("YXT", 0, 1), ("YXT", 1, 0), ("YXT", 1, 1), ("XYT", 0, 0), ("XYT", 0, 1), ("XYT", 1, 0), ("XYT", 1, 1)]
DONE_FIRST = [0]
DILATIONS = [1, 1]
HEAD_K = 3
BCAST = False
DENSE = False
WINPOOL = False
STEM = 64
WIDTH = 40
HEADC = 32
GLOB = 64
MASK_EMB = 16
DENSE_C = 8
DENSE_H = 256
WP = 16
WINDOWS = [(1, 2, 0), (1, 2, 1), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1)]


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


class WinPool(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj = nn.Conv2d(c, WP, 1)
        self.mix = nn.Conv2d(WP * (1 + 3 * len(WINDOWS)), c, 1)

    def forward(self, h):
        p = self.proj(h)
        parts = [p]
        for dy, dx, br in WINDOWS:
            if br:
                q = F.pad(p, (dx - 1, 0, dy - 1, 0))
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts.append(F.max_pool2d(q, (dy, dx), stride=1))
            parts.append(-F.max_pool2d(-q, (dy, dx), stride=1))
            parts.append(F.avg_pool2d(q, (dy, dx), stride=1))
        return F.relu(self.mix(torch.cat(parts, 1)))


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
        self.combos = [(d, lay) for d in DONE_FIRST for lay in LAYOUTS]
        if self.grid_ok:
            n_mask = len(self.combos) * self.nt
        else:
            n_mask = MASK_EMB
            self.mask_emb = nn.Linear(self.n_actions, MASK_EMB * self.ny * self.nx)
        cin = len(OBS_VIEWS) * self.cin_obs + n_mask + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILATIONS])
        self.hk = nn.Conv2d(WIDTH, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEADC, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, len(DONE_FIRST))
        self.value_head = nn.Linear(GLOB, 1)
        if BCAST:
            self.bc = nn.Linear(GLOB, HEADC)
        if DENSE:
            self.dred = nn.Conv2d(WIDTH, DENSE_C, 1)
            self.dfc = nn.Linear(DENSE_C * self.ny * self.nx + self.n_actions, DENSE_H)
            self.dbc = nn.Linear(DENSE_H, HEADC)
            self.dout = nn.Linear(DENSE_H, self.n_actions)
        if WINPOOL:
            self.wp = WinPool(HEADC)
        if self.grid_ok:
            self.outs = nn.Conv2d(HEADC, len(self.combos) * self.nt, 1)
        else:
            self.fc_place = nn.Linear(HEADC * self.ny * self.nx, self.n_place)

    def forward(self, obs, mask):
        b = obs.shape[0]
        ny, nx, nt, npl = self.ny, self.nx, self.nt, self.n_place
        so = {"C": self.cin_obs, "Y": ny, "X": nx}
        ins = [read_grid(obs, o, "CYX", so, fy, fx) for o, fy, fx in OBS_VIEWS]
        m = mask.to(obs.dtype)
        if self.grid_ok:
            sm = {"T": nt, "Y": ny, "X": nx}
            for d, lay in self.combos:
                o, fy, fx = lay
                if d:
                    mp = m[:, 1:1 + npl]
                else:
                    mp = m[:, :npl]
                ins.append(read_grid(mp, o, "TYX", sm, fy, fx))
        else:
            ins.append(self.mask_emb(m).view(b, MASK_EMB, ny, nx))
        cy = torch.linspace(-1.0, 1.0, ny, device=obs.device, dtype=obs.dtype).view(1, 1, ny, 1).expand(b, 1, ny, nx)
        cx = torch.linspace(-1.0, 1.0, nx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, nx).expand(b, 1, ny, nx)
        ins.append(cy)
        ins.append(cx)
        f = self.stem(torch.cat(ins, 1))
        for conv in self.res:
            f = F.relu(f + conv(f))
        h = F.relu(self.hk(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        if BCAST:
            h = F.relu(h + self.bc(g)[:, :, None, None])
        dh = None
        if DENSE:
            dh = F.relu(self.dfc(torch.cat([F.relu(self.dred(f)).flatten(1), m], 1)))
            h = F.relu(h + self.dbc(dh)[:, :, None, None])
        if WINPOOL:
            h = h + self.wp(h)
        dn = self.done_head(g)
        if self.grid_ok:
            maps = self.outs(h)
            logits = None
            for j, d in enumerate(DONE_FIRST):
                pl = None
                for i, (dd, lay) in enumerate(self.combos):
                    if dd != d:
                        continue
                    o, fy, fx = lay
                    part = write_grid(maps[:, i * nt:(i + 1) * nt], o, "TYX", fy, fx)
                    pl = part if pl is None else pl + part
                dj = dn[:, j:j + 1]
                if d:
                    full = torch.cat([dj, pl], 1)
                else:
                    full = torch.cat([pl, dj], 1)
                logits = full if logits is None else logits + full
        else:
            logits = torch.cat([self.fc_place(h.flatten(1)), dn[:, :1]], 1)
        if DENSE:
            logits = logits + self.dout(dh)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
