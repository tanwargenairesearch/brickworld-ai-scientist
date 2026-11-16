import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
           (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
           (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1),
           (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
OBS_READS = [0]
DENSE = False
ATTN = True
SEP_CRITIC = False
NORM = False
FILM = False
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
EMB = 8


def to_grid(v, lay, T, Y, X):
    o, tr, fy, fx = lay
    B = v.shape[0]
    if o == 0:
        g = v.reshape(B, T, X, Y).transpose(2, 3) if tr else v.reshape(B, T, Y, X)
    else:
        g = v.reshape(B, X, Y, T).permute(0, 3, 2, 1) if tr else v.reshape(B, Y, X, T).permute(0, 3, 1, 2)
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def from_grid(g, lay):
    o, tr, fy, fx = lay
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    if o == 0:
        if tr:
            g = g.transpose(2, 3)
    else:
        g = g.permute(0, 3, 2, 1) if tr else g.permute(0, 2, 3, 1)
    return g.reshape(g.shape[0], -1)


def read_obs(obs, mode, C, Y, X):
    B = obs.shape[0]
    if mode == 1:
        return obs.reshape(B, Y, X, C).permute(0, 3, 1, 2)
    if mode == 2:
        return obs.reshape(B, X, Y, C).permute(0, 3, 2, 1)
    return obs


def norm(c):
    if NORM:
        return nn.GroupNorm(8 if c % 8 == 0 else 1, c)
    return nn.Identity()


class Attn(nn.Module):
    def __init__(self, c, n, heads):
        super().__init__()
        self.heads = heads
        self.ln = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        self.pos = nn.Parameter(torch.zeros(1, n, c))
        nn.init.normal_(self.pos, std=0.02)
        self.ln2 = nn.LayerNorm(c)
        self.ff = nn.Sequential(nn.Linear(c, 2 * c), nn.ReLU(), nn.Linear(2 * c, c))

    def forward(self, x):
        B, C, Y, X = x.shape
        t = x.flatten(2).transpose(1, 2) + self.pos
        q, k, v = self.qkv(self.ln(t)).chunk(3, -1)
        H = self.heads
        D = C // H
        q = q.reshape(B, -1, H, D).transpose(1, 2)
        k = k.reshape(B, -1, H, D).transpose(1, 2)
        v = v.reshape(B, -1, H, D).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D), -1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, -1, C)
        t = t + self.proj(o)
        t = t + self.ff(self.ln2(t))
        return t.transpose(1, 2).reshape(B, C, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.ok = self.n_place == self.T * self.Y * self.X
        self.lays = list(LAYOUTS) if self.ok else []
        self.reads = list(OBS_READS)
        self.use_memb = DENSE or not self.ok
        self.use_fl = DENSE or not self.ok
        YX = self.Y * self.X
        cin = self.C * len(self.reads) + 2 + self.T * len(self.lays)
        if self.use_memb:
            self.memb = nn.Linear(self.n_place, EMB * YX)
            cin += EMB
        if DENSE:
            self.oemb = nn.Linear(self.C * YX, (EMB // 2) * YX)
            cin += EMB // 2
        self.s1 = nn.Conv2d(cin, STEM, 1)
        self.sn = norm(STEM)
        self.s2 = nn.Conv2d(STEM, WIDTH, 1)
        if FILM:
            self.fconv = nn.Conv2d(self.C, 16, 1)
            self.film = nn.Linear(16, 2 * WIDTH)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n1 = norm(WIDTH)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = norm(WIDTH)
        if ATTN:
            self.attn = Attn(WIDTH, YX, 4)
        self.hk = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.hs = nn.Conv2d(WIDTH, HEAD, 1)
        if self.ok:
            self.out = nn.Conv2d(HEAD, self.T * len(self.lays), 1)
        if self.use_fl:
            self.fo = nn.Conv2d(HEAD, 4, 1)
            self.fl = nn.Linear(4 * YX, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done = nn.Linear(GH, 1)
        if SEP_CRITIC:
            self.cc1 = nn.Conv2d(self.C + 2, 32, 1)
            self.cn = norm(32)
            self.cc2 = nn.Conv2d(32, 32, 3, padding=1)
            self.cv = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        else:
            self.value = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T, C = self.Y, self.X, self.T, self.C
        m = mask.float()
        mp = m[:, :self.n_place]
        feats = [read_obs(obs, r, C, Y, X) for r in self.reads]
        for lay in self.lays:
            feats.append(to_grid(mp, lay, T, Y, X))
        if self.use_memb:
            feats.append(self.memb(mp).view(B, EMB, Y, X))
        if DENSE:
            feats.append(self.oemb(obs.reshape(B, -1)).view(B, EMB // 2, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        x = F.relu(self.sn(self.s1(torch.cat(feats, 1))))
        x = F.relu(self.s2(x))
        if FILM:
            p = F.relu(self.fconv(obs)).mean((2, 3))
            ga, be = self.film(p).chunk(2, 1)
            x = x * (1.0 + ga[:, :, None, None]) + be[:, :, None, None]
        x = F.relu(x + self.n1(self.r1(x)))
        x = F.relu(x + self.n2(self.r2(x)))
        if ATTN:
            x = self.attn(x)
        h = F.relu(self.hk(x) + self.hs(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        place = 0
        if self.ok:
            o = self.out(h)
            for i, lay in enumerate(self.lays):
                place = place + from_grid(o[:, i * T:(i + 1) * T], lay)
        if self.use_fl:
            place = place + self.fl(F.relu(self.fo(h)).flatten(1))
        logits = torch.cat([place, self.done(g)], 1)
        if SEP_CRITIC:
            z = F.relu(self.cn(self.cc1(torch.cat([obs, yy, xx], 1))))
            z = F.relu(self.cc2(z))
            value = self.cv(torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1)).squeeze(1)
        else:
            value = self.value(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
