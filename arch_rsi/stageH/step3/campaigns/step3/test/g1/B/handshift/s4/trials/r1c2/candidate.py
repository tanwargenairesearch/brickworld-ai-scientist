import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEADC = 32
HEAD_K = 3
LAYOUTS = [
    ("major", 0, 0, 0), ("major", 0, 0, 1), ("major", 0, 1, 0), ("major", 0, 1, 1),
    ("major", 1, 0, 0), ("major", 1, 0, 1), ("major", 1, 1, 0), ("major", 1, 1, 1),
    ("minor", 0, 0, 0), ("minor", 0, 0, 1), ("minor", 0, 1, 0), ("minor", 0, 1, 1),
    ("minor", 1, 0, 0), ("minor", 1, 0, 1), ("minor", 1, 1, 0), ("minor", 1, 1, 1),
]
OBS_SYMS = [(0, 0, 0)]
OBS_CLAST = False
DONE_HEDGE = False


def to_canon(m, lay, T, Y, X):
    B = m.shape[0]
    order, tr, fy, fx = lay
    if order == "major":
        if tr:
            v = m.reshape(B, T, X, Y).transpose(2, 3)
        else:
            v = m.reshape(B, T, Y, X)
    else:
        if tr:
            v = m.reshape(B, X, Y, T).permute(0, 3, 2, 1)
        else:
            v = m.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    if fy:
        v = v.flip(2)
    if fx:
        v = v.flip(3)
    return v


def from_canon(o, lay):
    B = o.shape[0]
    order, tr, fy, fx = lay
    if fy:
        o = o.flip(2)
    if fx:
        o = o.flip(3)
    if order == "major":
        if tr:
            o = o.transpose(2, 3)
    else:
        if tr:
            o = o.permute(0, 3, 2, 1)
        else:
            o = o.permute(0, 2, 3, 1)
    return o.reshape(B, -1)


def sym(x, s):
    tr, fy, fx = s
    if fy:
        x = x.flip(2)
    if fx:
        x = x.flip(3)
    if tr:
        x = x.transpose(2, 3)
    return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T, self.Y, self.X = spec.n_types, spec.Y, spec.X
        self.C = spec.in_channels
        self.n_place = spec.n_place
        self.ok = spec.n_place == self.T * self.Y * self.X
        self.lays = list(LAYOUTS) if self.ok else []
        self.syms = [s for s in OBS_SYMS if s[0] == 0 or self.X == self.Y]
        self.done_hedge = DONE_HEDGE and self.ok
        if self.ok:
            nm = len(self.lays) * self.T + (self.T if self.done_hedge else 0)
        else:
            self.mask_emb = nn.Linear(spec.n_actions, 4 * self.Y * self.X)
            nm = 4
        cin = len(self.syms) * self.C + (self.C if OBS_CLAST else 0) + nm + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hc = nn.Conv2d(WIDTH, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hs = nn.Conv2d(WIDTH, HEADC, 1)
        if self.ok:
            nout = len(self.lays) * self.T + (self.T if self.done_hedge else 0)
            self.out = nn.Conv2d(HEADC, nout, 1)
        else:
            self.out = nn.Conv2d(HEADC, 4, 1)
            self.fb = nn.Linear(4 * self.Y * self.X, spec.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done = nn.Linear(64, 1)
        self.done2 = nn.Linear(64, 1)
        self.value = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask.float()
        parts = [sym(obs, s) for s in self.syms]
        if OBS_CLAST:
            parts.append(obs.reshape(B, Y, X, self.C).permute(0, 3, 1, 2))
        if self.ok:
            mp = m[:, :self.n_place]
            for lay in self.lays:
                parts.append(to_canon(mp, lay, T, Y, X))
            if self.done_hedge:
                parts.append(to_canon(m[:, 1:1 + self.n_place], ("major", 0, 0, 0), T, Y, X))
        else:
            parts.append(self.mask_emb(m).view(B, 4, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts.append(yy.to(obs.dtype))
        parts.append(xx.to(obs.dtype))
        x = self.stem(torch.cat(parts, dim=1))
        x = F.relu(x + self.c1(x))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.hc(f) + self.hs(f))
        o = self.out(h)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        if self.ok:
            place = 0
            for k, lay in enumerate(self.lays):
                place = place + from_canon(o[:, k * T:(k + 1) * T], lay)
            logits = torch.cat([place, self.done(g)], dim=1)
            if self.done_hedge:
                p2 = from_canon(o[:, len(self.lays) * T:], ("major", 0, 0, 0))
                logits = logits + torch.cat([self.done2(g), p2], dim=1)
        else:
            logits = torch.cat([self.fb(o.flatten(1)), self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
