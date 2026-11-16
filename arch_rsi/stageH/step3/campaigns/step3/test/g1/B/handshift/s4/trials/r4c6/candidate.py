import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.003, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEADC = 32
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
MASK_DENSE = True
OBS_DENSE = True
DENSE_OUT = True
NORM = True
VGRAD = 0.05


def to_canon(m, lay, T, Y, X):
    B = m.shape[0]
    tr, fy, fx = lay
    if tr:
        v = m.reshape(B, T, X, Y).transpose(2, 3)
    else:
        v = m.reshape(B, T, Y, X)
    if fy:
        v = v.flip(2)
    if fx:
        v = v.flip(3)
    return v


def from_canon(o, lay):
    B = o.shape[0]
    tr, fy, fx = lay
    if fy:
        o = o.flip(2)
    if fx:
        o = o.flip(3)
    if tr:
        o = o.transpose(2, 3)
    return o.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T, Y, X = int(spec.n_types), int(spec.Y), int(spec.X)
        self.T, self.Y, self.X = T, Y, X
        self.C = int(spec.in_channels)
        self.P = int(spec.n_place)
        self.NA = int(spec.n_actions)
        self.n_extra = self.NA - self.P
        self.ok = self.P == T * Y * X
        self.lays = list(LAYOUTS) if self.ok else []
        N = Y * X
        cin = self.C + 2 + len(self.lays) * T
        self.km = 0
        if MASK_DENSE or not self.ok:
            self.km = max(1, min(4, 200000 // max(1, self.NA * N)))
            self.mask_emb = nn.Linear(self.NA, self.km * N)
            cin += self.km
        self.ko = 0
        if OBS_DENSE:
            self.ko = max(1, min(8, 632 // N))
            self.oproj = nn.Conv2d(self.C, self.ko, 1)
            self.oremap = nn.Linear(self.ko * N, self.ko * N)
            cin += self.ko
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.norm = nn.GroupNorm(1, STEM) if NORM else None
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hc = nn.Conv2d(WIDTH, HEADC, 3, padding=1)
        self.hs = nn.Conv2d(WIDTH, HEADC, 1)
        if self.ok:
            self.out = nn.Conv2d(HEADC, len(self.lays) * T, 1)
        self.kd = 0
        if DENSE_OUT or not self.ok:
            self.kd = max(1, min(8, 300000 // max(1, N * self.P)))
            self.dproj = nn.Conv2d(HEADC, self.kd, 1)
            self.dense = nn.Linear(self.kd * N, self.P)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done = nn.Linear(64, max(1, self.n_extra))
        self.value = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask.to(obs.dtype)
        parts = [obs]
        if self.ok:
            mp = m[:, :self.P]
            for lay in self.lays:
                parts.append(to_canon(mp, lay, T, Y, X))
        if self.km > 0:
            parts.append(self.mask_emb(m).reshape(B, self.km, Y, X))
        if self.ko > 0:
            q = self.oproj(obs).reshape(B, -1)
            parts.append(self.oremap(q).reshape(B, self.ko, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(yy)
        parts.append(xx)
        x = self.stem1(torch.cat(parts, dim=1))
        if self.norm is not None:
            x = self.norm(x)
        x = F.relu(x)
        x = F.relu(self.stem2(x))
        x = F.relu(x + self.c1(x))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.hc(f) + self.hs(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = None
        if self.ok:
            o = self.out(h)
            for k, lay in enumerate(self.lays):
                p = from_canon(o[:, k * T:(k + 1) * T], lay)
                place = p if place is None else place + p
        if self.kd > 0:
            p = self.dense(self.dproj(h).reshape(B, -1))
            place = p if place is None else place + p
        if self.n_extra > 0:
            logits = torch.cat([place, self.done(g)], dim=1)
        else:
            logits = place
        v = self.value(g).squeeze(1)
        if VGRAD < 1.0:
            v = VGRAD * v + (1.0 - VGRAD) * v.detach()
        return logits, v


def build_policy(spec):
    return Net(spec)
