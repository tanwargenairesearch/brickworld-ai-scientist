import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEAD = 32
HK = 3
WIDE = False
OFFSETS = (0, 1)
OBS_VIEWS = False
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X, T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.Y, self.X, self.T = Y, X, T
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.ok = (self.n_place == T * Y * X) and (self.n_actions == self.n_place + 1)
        self.nview = (4 if Y == X else 3) if OBS_VIEWS else 1
        cin = int(spec.in_channels) * self.nview + 2
        if self.ok:
            gi, si, offs = [], [], []
            for o in OFFSETS:
                for lay in layouts():
                    idx = layout_index(T, Y, X, lay[0], lay[1], lay[2], lay[3])
                    inv = [0] * len(idx)
                    for p, a in enumerate(idx):
                        inv[a] = p
                    gi.append([a + o for a in idx])
                    si.append(inv)
                    offs.append(o)
            self.register_buffer("gidx", torch.tensor(gi, dtype=torch.long).flatten())
            self.register_buffer("sidx", torch.tensor(si, dtype=torch.long))
            self.offs = offs
            self.L = len(offs)
            cin += self.L * T
            nout = self.L * T
        else:
            self.offs = []
            self.L = 0
            self.membed = nn.Linear(self.n_actions, MEMB * Y * X)
            cin += MEMB
            nout = MEMB
            self.fallback = nn.Linear(MEMB * Y * X, self.n_actions - 1)
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HEAD, HK, padding=HK // 2)
        self.hskip = nn.Conv2d(WIDTH, HEAD, 1)
        self.hout = nn.Conv2d(HEAD, nout, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        views = [obs]
        if self.nview > 1:
            views.append(obs.flip(-2))
            views.append(obs.flip(-1))
            if self.nview > 3:
                views.append(obs.transpose(-1, -2))
        m = mask.float()
        if self.ok:
            mi = m[:, self.gidx].reshape(B, self.L * T, Y, X)
        else:
            mi = self.membed(m).reshape(B, MEMB, Y, X)
        cy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        cx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat(views + [mi, cy, cx], 1))
        h = F.relu(h + self.r1(h))
        f = F.relu(h + self.r2(h))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        hd = F.relu(self.hconv(f) + self.hskip(f))
        out = self.hout(hd)
        if self.ok:
            out = out.reshape(B, self.L, T * Y * X)
            logits = F.pad(done, (self.n_place, 0))
            if 1 in self.offs:
                logits = logits + F.pad(done, (0, self.n_place))
            for k in range(self.L):
                o = self.offs[k]
                part = out[:, k].index_select(1, self.sidx[k])
                logits = logits + F.pad(part, (o, self.n_actions - self.n_place - o))
        else:
            logits = torch.cat([self.fallback(out.flatten(1)), done], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
