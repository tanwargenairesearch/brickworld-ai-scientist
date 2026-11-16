import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.9}

STEMW = 48
COLW = 32
NRES = 1
HEADC = 32
LW = 16
POOLC = 24
ATTD = 32
LAYOUTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
USE_LAYER = True
USE_POOL = True
USE_ATTN = False


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


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, proj, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(proj, 1, 1) for _ in self.fps])

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
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        return torch.cat(res, dim=1)


class GlobalAttn(nn.Module):
    def __init__(self, cin_obs, cq, d, n):
        super().__init__()
        self.kpos = nn.Embedding(n, d)
        self.qpos = nn.Embedding(n, d)
        self.kin = nn.Conv2d(cin_obs, d, 1)
        self.qin = nn.Conv2d(cq, d, 1)
        self.k = nn.Linear(d, d)
        self.q = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, cq)
        self.d = d

    def forward(self, obs, c):
        B, _, Y, X = c.shape
        idx = torch.arange(Y * X, device=c.device)
        ko = self.kin(obs).flatten(2).transpose(1, 2) + self.kpos(idx).unsqueeze(0)
        qo = self.qin(c).flatten(2).transpose(1, 2) + self.qpos(idx).unsqueeze(0)
        a = torch.softmax(self.q(qo) @ self.k(ko).transpose(1, 2) / math.sqrt(self.d), dim=-1)
        out = self.o(a @ self.v(ko))
        return out.transpose(1, 2).reshape(B, -1, Y, X)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T, Y, X, Z = int(spec.n_types), int(spec.Y), int(spec.X), int(spec.Z)
        C = int(spec.in_channels)
        self.T, self.Y, self.X, self.Z, self.C = T, Y, X, Z, C
        self.P = int(spec.n_place)
        self.NA = int(spec.n_actions)
        self.ok = self.P == T * Y * X
        self.lays = list(LAYOUTS) if self.ok else []
        if self.ok:
            nk = len(self.lays) * T
        else:
            self.mask_emb = nn.Linear(self.NA, 4 * Y * X)
            nk = 4
        cin = C + nk + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEMW, 1), nn.ReLU(), nn.Conv2d(STEMW, COLW, 1), nn.ReLU())
        self.use_attn = USE_ATTN
        if self.use_attn:
            self.attn = GlobalAttn(C, COLW, ATTD, Y * X)
        self.use_layer = USE_LAYER and C == 4 * Z + 3 and Z >= 1
        if self.use_layer:
            self.lin = nn.Conv2d(17 + Z, LW, 1)
            self.lall = nn.Conv2d(C, Z * LW, 1)
            self.lcol = nn.Conv2d(COLW, LW, 1)
            self.lc1 = nn.Conv2d(LW, LW, 3, padding=1)
            self.lc2 = nn.Conv2d(LW, LW, 3, padding=1)
            self.lout = nn.Conv2d(LW, nk, 1)
            self.merge = nn.Conv2d(2 * LW, COLW, 1)
        self.res = nn.ModuleList([nn.Conv2d(COLW, COLW, 3, padding=1) for _ in range(NRES)])
        self.hc = nn.Conv2d(COLW, HEADC, 3, padding=1)
        self.hs = nn.Conv2d(COLW, HEADC, 1)
        self.out = nn.Conv2d(HEADC, nk, 1)
        if not self.ok:
            self.fb = nn.Linear(4 * Y * X, self.P)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.use_pool = USE_POOL and self.ok and len(fps) == T and all(1 <= a <= Y and 1 <= b <= X for a, b in fps)
        if self.use_pool:
            self.pool = PoolHead(COLW, fps, POOLC)
        self.glob = nn.Sequential(nn.Linear(2 * COLW, 64), nn.ReLU())
        self.n_extra = self.NA - self.P
        self.done = nn.Linear(64, max(1, self.n_extra))
        self.value = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X, Z = self.T, self.Y, self.X, self.Z
        m = mask.to(obs.dtype)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        parts = [obs]
        if self.ok:
            mp = m[:, :self.P]
            for lay in self.lays:
                parts.append(to_canon(mp, lay, T, Y, X))
        else:
            parts.append(self.mask_emb(m).reshape(B, 4, Y, X))
        parts.append(yy)
        parts.append(xx)
        c = self.stem(torch.cat(parts, dim=1))
        if self.use_attn:
            c = F.relu(c + self.attn(obs, c))
        o_layer = None
        if self.use_layer:
            bld = obs[:, 0:Z]
            tgt = obs[:, Z:2 * Z]
            mis = obs[:, 2 * Z:3 * Z]
            ext = obs[:, 3 * Z:4 * Z]
            cur = torch.stack([bld, tgt, mis, ext], dim=2)
            zero = torch.zeros_like(cur[:, :1])
            below = torch.cat([zero, cur[:, :-1]], dim=1)
            above = torch.cat([cur[:, 1:], zero], dim=1)
            colc = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
            zid = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
            co = torch.cat([yy, xx], dim=1).unsqueeze(1).expand(B, Z, 2, Y, X)
            li = torch.cat([cur, below, above, colc, zid, co], dim=2).reshape(B * Z, 17 + Z, Y, X)
            pa = self.lall(obs).reshape(B * Z, LW, Y, X)
            pc = self.lcol(c).unsqueeze(1).expand(B, Z, LW, Y, X).reshape(B * Z, LW, Y, X)
            p = F.relu(self.lin(li) + pa + pc)
            p = F.relu(p + self.lc1(p))
            p = F.relu(p + self.lc2(p))
            lo = self.lout(p)
            o_layer = torch.logsumexp(lo.reshape(B, Z, lo.shape[1], Y, X), dim=1)
            p5 = p.reshape(B, Z, LW, Y, X)
            c = F.relu(c + self.merge(torch.cat([p5.amax(1), p5.mean(1)], dim=1)))
        for conv in self.res:
            c = F.relu(c + conv(c))
        f = c
        h = F.relu(self.hc(f) + self.hs(f))
        o = self.out(h)
        if o_layer is not None:
            o = o + o_layer
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        if self.ok:
            place = 0
            for k, lay in enumerate(self.lays):
                place = place + from_canon(o[:, k * T:(k + 1) * T], lay)
            if self.use_pool:
                place = place + self.pool(f).reshape(B, -1)
        else:
            place = self.fb(o.flatten(1))
        if self.n_extra > 0:
            logits = torch.cat([place, self.done(g)], dim=1)
        else:
            logits = place
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
