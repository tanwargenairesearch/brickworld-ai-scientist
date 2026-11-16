import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

ORDERS = ("tyx", "txy", "ytx", "xty", "yxt", "xyt")
HEAD_K = 3
POOL = True
CONTEXT = True
NORM = True
ATTN = False
DILS = (1, 1)
STEM = 64
W = 40
HP = 32
PP = 8
HM = 48


def make_layouts():
    return [(o, fy, fx) for o in ORDERS for fy in (False, True) for fx in (False, True)]


def flip_dims(fy, fx):
    d = []
    if fy:
        d.append(2)
    if fx:
        d.append(3)
    return d


def to_grid(m, lay, T, Y, X):
    order, fy, fx = lay
    size = {"t": T, "y": Y, "x": X}
    g = m.reshape(m.shape[0], size[order[0]], size[order[1]], size[order[2]])
    g = g.permute(0, 1 + order.index("t"), 1 + order.index("y"), 1 + order.index("x"))
    d = flip_dims(fy, fx)
    if d:
        g = g.flip(d)
    return g


def from_grid(g, lay):
    order, fy, fx = lay
    d = flip_dims(fy, fx)
    if d:
        g = g.flip(d)
    g = g.permute(0, 1 + "tyx".index(order[0]), 1 + "tyx".index(order[1]), 1 + "tyx".index(order[2]))
    return g.reshape(g.shape[0], -1)


def corner_pool(p, dy, dx, cy, cx):
    px = (dx - 1, 0) if cx else (0, dx - 1)
    py = (dy - 1, 0) if cy else (0, dy - 1)
    q = F.pad(p, (px[0], px[1], py[0], py[1]))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    return mx, mn


class Block(nn.Module):
    def __init__(self, w, dil, context):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=dil, dilation=dil)
        self.context = context
        if context:
            self.glin = nn.Linear(2 * w, w)
            self.rc = nn.Conv2d(2 * w, w, 1)

    def forward(self, x):
        y = self.conv(x)
        if self.context:
            g = self.glin(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
            y = y + g[:, :, None, None]
            r = x.mean(3, keepdim=True).expand_as(x)
            c = x.mean(2, keepdim=True).expand_as(x)
            y = y + self.rc(torch.cat([r, c], dim=1))
        return F.relu(x + y)


class Attn(nn.Module):
    def __init__(self, w, Y, X, heads=4):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, w, Y, X) * 0.02)
        self.qkv = nn.Linear(w, 3 * w)
        self.out = nn.Linear(w, w)
        self.heads = heads

    def split(self, z, B, C):
        return z.reshape(B, -1, self.heads, C // self.heads).transpose(1, 2)

    def forward(self, x):
        B, C, Y, X = x.shape
        t = (x + self.pos).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(t).chunk(3, dim=-1)
        q, k, v = self.split(q, B, C), self.split(k, B, C), self.split(v, B, C)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(C // self.heads), dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, Y * X, C)
        o = self.out(o).transpose(1, 2).reshape(B, C, Y, X)
        return F.relu(x + o)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = spec.Y, spec.X, spec.n_types
        self.n_place = spec.n_place
        self.n_extra = spec.n_actions - spec.n_place
        self.ok = spec.n_place == self.T * self.Y * self.X
        self.lays = make_layouts()
        cin = spec.in_channels + 2
        if self.ok:
            cin += self.T * len(self.lays)
        else:
            self.mask_emb = nn.Linear(spec.n_actions, 4 * self.Y * self.X)
            cin += 4
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        if NORM:
            self.norm = nn.GroupNorm(1, STEM)
        self.stem2 = nn.Conv2d(STEM, W, 1)
        self.blocks = nn.ModuleList([Block(W, d, CONTEXT) for d in DILS])
        if ATTN:
            self.attn = Attn(W, self.Y, self.X)
        self.hconv = nn.Conv2d(W, HP, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(W, HP, 1)
        hc = HP
        if POOL:
            shapes = set()
            for a, b in spec.footprints:
                a, b = int(a), int(b)
                if a * b > 1:
                    shapes.add((a, b))
                    shapes.add((b, a))
            self.shapes = sorted(shapes)
            self.proj = nn.Conv2d(W, PP, 1)
            self.mix = nn.Conv2d(HP + PP + 8 * PP * len(self.shapes), HM, 1)
            hc = HM
        if self.ok:
            self.outs = nn.Conv2d(hc, self.T * len(self.lays), 1)
        else:
            self.red = nn.Conv2d(hc, 8, 1)
            self.dense = nn.Linear(8 * self.Y * self.X, spec.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, self.n_extra)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        ins = [obs, yy, xx]
        mf = mask.to(obs.dtype)
        if self.ok:
            mp = mf[:, :self.n_place]
            for lay in self.lays:
                ins.append(to_grid(mp, lay, T, Y, X))
        else:
            ins.append(self.mask_emb(mf).view(B, 4, Y, X))
        s = self.stem1(torch.cat(ins, dim=1))
        if NORM:
            s = self.norm(s)
        x = F.relu(self.stem2(F.relu(s)))
        for blk in self.blocks:
            x = blk(x)
        if ATTN:
            x = self.attn(x)
        f = x
        h = F.relu(self.hconv(f) + self.hskip(f))
        if POOL:
            p = self.proj(f)
            feats = [h, p]
            for dy, dx in self.shapes:
                for cy in (0, 1):
                    for cx in (0, 1):
                        mx, mn = corner_pool(p, dy, dx, cy, cx)
                        feats.append(mx)
                        feats.append(mn)
            h = F.relu(self.mix(torch.cat(feats, dim=1)))
        if self.ok:
            maps = self.outs(h)
            place = None
            for k, lay in enumerate(self.lays):
                pk = from_grid(maps[:, k * T:(k + 1) * T], lay)
                place = pk if place is None else place + pk
        else:
            place = self.dense(F.relu(self.red(h)).flatten(1))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
