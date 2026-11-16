import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.95, "gae_lambda": 0.9}

STEM = 64
WIDTH = 40
HEAD = 32
WIDE = False
NORM = True
SMALL_OUT = True
SEP_CRITIC = True
CRIT_W = 24
DILS = (1, 1)
GBCAST = False
FPOOL = False
PP = 8
MEMB = 8


def layouts():
    res = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
    if WIDE:
        res = []
        for tm in (0, 1):
            for tr in (0, 1):
                for fy in (0, 1):
                    for fx in (0, 1):
                        res.append((tr, fy, fx, tm))
    return res


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


def windows():
    res = []
    for dy, dx in ((1, 2), (2, 1), (2, 2)):
        for oy in range(dy):
            for ox in range(dx):
                res.append((dy, dx, oy, ox))
    return res


def norm(c):
    if NORM:
        return nn.GroupNorm(4, c)
    return nn.Identity()


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X, T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.Y, self.X, self.T = Y, X, T
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.ok = (self.n_place == T * Y * X) and (self.n_actions == self.n_place + 1)
        cin = int(spec.in_channels) + 2
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
            self.fallback = nn.Linear(MEMB * Y * X, self.n_actions - 1)
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.n1 = norm(STEM)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.n2 = norm(WIDTH)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILS])
        self.rn = nn.ModuleList([norm(WIDTH) for _ in DILS])
        if GBCAST:
            self.gb = nn.ModuleList([nn.Linear(WIDTH, WIDTH) for _ in DILS])
        self.hconv = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.hskip = nn.Conv2d(WIDTH, HEAD, 1)
        self.wins = windows()
        if FPOOL:
            self.pproj = nn.Conv2d(WIDTH, PP, 1)
            self.pmix = nn.Conv2d(PP * 2 * len(self.wins), HEAD, 1)
        self.hout = nn.Conv2d(HEAD, nout, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        if SEP_CRITIC:
            self.cstem = nn.Conv2d(cin, CRIT_W, 1)
            self.cc1 = nn.Conv2d(CRIT_W, CRIT_W, 3, padding=1)
            self.cc2 = nn.Conv2d(CRIT_W, CRIT_W, 3, padding=2, dilation=2)
            self.cmlp = nn.Sequential(nn.Linear(2 * CRIT_W, 64), nn.ReLU(), nn.Linear(64, 1))
        if SMALL_OUT:
            nn.init.normal_(self.hout.weight, std=0.01)
            nn.init.zeros_(self.hout.bias)
            nn.init.normal_(self.done_head.weight, std=0.01)
            nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        m = mask.to(obs.dtype)
        if self.ok:
            mi = m[:, self.gidx].reshape(B, self.L * T, Y, X)
        else:
            mi = self.membed(m).reshape(B, MEMB, Y, X)
        cy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        cx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = torch.cat([obs, mi, cy, cx], 1)
        h = F.relu(self.n1(self.stem1(x)))
        h = F.relu(self.n2(self.stem2(h)))
        for i in range(len(self.res)):
            u = self.res[i](h)
            if GBCAST:
                u = u + self.gb[i](h.mean((2, 3))).unsqueeze(-1).unsqueeze(-1)
            h = F.relu(h + self.rn[i](u))
        f = h
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        hd = self.hconv(f) + self.hskip(f)
        if FPOOL:
            p = F.relu(self.pproj(f))
            pools = []
            for dy, dx, oy, ox in self.wins:
                q = F.pad(p, (ox, dx - 1 - ox, oy, dy - 1 - oy))
                pools.append(F.max_pool2d(q, (dy, dx), stride=1))
                pools.append(-F.max_pool2d(-q, (dy, dx), stride=1))
            hd = hd + self.pmix(torch.cat(pools, 1))
        hd = F.relu(hd)
        out = self.hout(hd)
        if self.ok:
            out = out.reshape(B, self.L, T * Y * X)
            place = out[:, 0].index_select(1, self.sidx[0])
            for k in range(1, self.L):
                place = place + out[:, k].index_select(1, self.sidx[k])
        else:
            place = self.fallback(out.flatten(1))
        logits = torch.cat([place, done], 1)
        if SEP_CRITIC:
            c = F.relu(self.cstem(x))
            c = F.relu(self.cc1(c))
            c = F.relu(c + self.cc2(c))
            value = self.cmlp(torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1)).squeeze(1)
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
