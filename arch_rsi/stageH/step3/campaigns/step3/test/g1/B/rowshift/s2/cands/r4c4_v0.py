import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HP = 32
HK = 3
MEMB = 8
LAYOUTS = [("TYX", ()), ("TXY", ()), ("TYX", (2,)), ("TYX", (3,))]
DILATE = [1, 1]
USE_MASK = False
POOLHEAD = True
ATTN = 0
HEADS = 4
GMLP = False
GMLP_MAX = 256
ORTHO = False


def to_grid(v, order, flips, T, Y, X):
    B = v.shape[0]
    if order == "TYX":
        g = v.reshape(B, T, Y, X)
    else:
        g = v.reshape(B, T, X, Y).transpose(2, 3)
    if len(flips) > 0:
        g = g.flip(list(flips))
    return g


def from_grid(m, order, flips):
    if len(flips) > 0:
        m = m.flip(list(flips))
    if order == "TYX":
        out = m
    else:
        out = m.transpose(2, 3)
    return out.reshape(m.shape[0], -1)


def coords(obs, Y, X):
    B = obs.shape[0]
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([ys, xs], dim=1)


class AttnBlock(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, D = x.shape
        dh = D // self.h
        t = self.qkv(self.ln1(x)).view(B, N, 3, self.h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = t.unbind(0)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.ln2(x))))
        return x


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
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.P = self.Y * self.X
        self.C = int(spec.in_channels)
        self.T = int(spec.n_types)
        self.W = WIDTH
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.grid_ok = (self.n_place == self.T * self.P) and (self.n_actions == self.n_place + 1)
        cin = self.C + 2
        if USE_MASK:
            if self.grid_ok:
                cin += len(LAYOUTS) * self.T
            else:
                self.mask_embed = nn.Linear(self.n_actions, MEMB * self.P)
                cin += MEMB
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILATE])
        self.attn = nn.ModuleList([AttnBlock(WIDTH, HEADS, 2 * WIDTH) for _ in range(ATTN)])
        if ATTN > 0:
            self.pos = nn.Embedding(self.P, WIDTH)
        self.hconv = nn.Conv2d(WIDTH, HP, HK, padding=HK // 2)
        self.hskip = nn.Conv2d(WIDTH, HP, 1)
        self.use_pool = POOLHEAD and self.grid_ok
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(HP, self.T, 1) for _ in LAYOUTS])
            if self.use_pool:
                self.pool = PoolHead(WIDTH, spec.footprints, 32, 32)
        else:
            self.out_proj = nn.Conv2d(HP, 4, 1)
            self.out_lin = nn.Linear(4 * self.P, self.n_actions - 1)
        if GMLP:
            nin = self.C * self.P + self.n_actions
            hid = max(32, min(GMLP_MAX, 400000 // (nin + self.n_actions)))
            self.g1 = nn.Linear(nin, hid)
            self.g2 = nn.Linear(hid, self.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        if ORTHO:
            self.init_ortho()

    def init_ortho(self):
        for mod in self.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(mod.weight, math.sqrt(2.0))
                nn.init.zeros_(mod.bias)
        pol = [self.done_head]
        if self.grid_ok:
            pol = pol + list(self.outs)
        else:
            pol = pol + [self.out_lin]
        for mod in pol:
            nn.init.orthogonal_(mod.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        m = mask.float()
        parts = [obs, coords(obs, Y, X)]
        if USE_MASK:
            if self.grid_ok:
                mp = m[:, :self.n_place]
                for o, fl in LAYOUTS:
                    parts.append(to_grid(mp, o, fl, T, Y, X))
            else:
                parts.append(self.mask_embed(m).view(B, MEMB, Y, X))
        x = F.relu(self.stem2(F.relu(self.stem1(torch.cat(parts, dim=1)))))
        for conv in self.res:
            x = F.relu(x + conv(x))
        if ATTN > 0:
            t = x.flatten(2).transpose(1, 2) + self.pos.weight.unsqueeze(0)
            for blk in self.attn:
                t = blk(t)
            x = t.transpose(1, 2).reshape(B, self.W, Y, X)
        f = x
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        if self.grid_ok:
            place = 0
            for k, (o, fl) in enumerate(LAYOUTS):
                place = place + from_grid(self.outs[k](h), o, fl)
            if self.use_pool:
                place = place + self.pool(f)
        else:
            place = self.out_lin(self.out_proj(h).flatten(1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if GMLP:
            z = torch.cat([obs.reshape(B, -1), m], dim=1)
            logits = logits + self.g2(F.relu(self.g1(z)))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
