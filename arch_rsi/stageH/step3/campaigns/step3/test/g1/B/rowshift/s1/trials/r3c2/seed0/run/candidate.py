import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEAD = 32
WIDE = True
DIHEDRAL = 1
CHLAST = True
OBS_DENSE = 0
MASK_DENSE = 0
MEMB = 8


def layouts():
    if WIDE:
        res = []
        for tm in (0, 1):
            for tr in (0, 1):
                for fy in (0, 1):
                    for fx in (0, 1):
                        res.append((tr, fy, fx, tm))
        return res
    return [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]


def layout_index(T, Y, X, tr, fy, fx, tm):
    idx = []
    for t in range(T):
        for y in range(Y):
            for x in range(X):
                yy = Y - 1 - y if fy else y
                xx = X - 1 - x if fx else x
                cell = xx * Y + yy if tr else yy * X + xx
                idx.append(cell * T + t if tm else t * Y * X + cell)
    return idx


def obs_views(Y, X):
    res = []
    for k in range(8):
        tr, fy, fx = k // 4, (k // 2) % 2, k % 2
        if tr and Y != X:
            continue
        res.append((tr, fy, fx))
    return res[:DIHEDRAL]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X, T = int(spec.Y), int(spec.X), int(spec.n_types)
        C = int(spec.in_channels)
        self.Y, self.X, self.T, self.C = Y, X, T, C
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.ok = (self.n_place == T * Y * X) and (self.n_actions == self.n_place + 1)
        self.dih = obs_views(Y, X)
        nv = len(self.dih) + (2 if CHLAST else 0)
        cin = C * nv + 2 + OBS_DENSE + MASK_DENSE
        if OBS_DENSE > 0:
            self.odense = nn.Linear(C * Y * X, OBS_DENSE * Y * X)
        if MASK_DENSE > 0:
            self.mdense = nn.Linear(self.n_actions, MASK_DENSE * Y * X)
            self.mdproj = nn.Conv2d(HEAD, MASK_DENSE, 1)
            self.mdout = nn.Linear(MASK_DENSE * Y * X, self.n_place)
        if self.ok:
            gi, si = [], []
            for lay in layouts():
                idx = layout_index(T, Y, X, lay[0], lay[1], lay[2], lay[3])
                inv = [0] * len(idx)
                for p, a in enumerate(idx):
                    inv[a] = p
                gi.append(idx)
                si.append(inv)
            self.L = len(gi)
            self.register_buffer("gidx", torch.tensor(gi, dtype=torch.long).flatten())
            self.register_buffer("sidx", torch.tensor(si, dtype=torch.long))
            cin += self.L * T
            nout = self.L * T
        else:
            self.L = 0
            self.membed = nn.Linear(self.n_actions, MEMB * Y * X)
            cin += MEMB
            nout = MEMB
            self.fallback = nn.Linear(MEMB * Y * X, self.n_place)
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.hskip = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, nout, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T, C = self.Y, self.X, self.T, self.C
        views = []
        for tr, fy, fx in self.dih:
            v = obs
            if fy:
                v = v.flip(-2)
            if fx:
                v = v.flip(-1)
            if tr:
                v = v.transpose(-1, -2)
            views.append(v)
        flat = obs.reshape(B, -1)
        if CHLAST:
            views.append(flat.reshape(B, Y, X, C).permute(0, 3, 1, 2))
            views.append(flat.reshape(B, X, Y, C).permute(0, 3, 2, 1))
        if OBS_DENSE > 0:
            views.append(self.odense(flat).reshape(B, OBS_DENSE, Y, X))
        m = mask.to(obs.dtype)
        if self.ok:
            views.append(m[:, self.gidx].reshape(B, self.L * T, Y, X))
        else:
            views.append(self.membed(m).reshape(B, MEMB, Y, X))
        if MASK_DENSE > 0:
            views.append(self.mdense(m).reshape(B, MASK_DENSE, Y, X))
        cy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        cx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        views.append(cy)
        views.append(cx)
        h = self.stem(torch.cat(views, 1))
        h = F.relu(h + self.r1(h))
        f = F.relu(h + self.r2(h))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        hd = F.relu(self.hconv(f) + self.hskip(f))
        out = self.hout(hd)
        if self.ok:
            out = out.reshape(B, self.L, T * Y * X)
            place = out[:, 0].index_select(1, self.sidx[0])
            for k in range(1, self.L):
                place = place + out[:, k].index_select(1, self.sidx[k])
        else:
            place = self.fallback(out.flatten(1))
        if MASK_DENSE > 0:
            place = place + self.mdout(self.mdproj(hd).flatten(1))
        logits = torch.cat([place, done], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
