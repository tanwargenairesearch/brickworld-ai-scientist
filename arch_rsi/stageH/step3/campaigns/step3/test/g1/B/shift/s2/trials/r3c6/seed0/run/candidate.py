import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 1, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = (("TYX", 0, 0), ("TYX", 1, 0), ("TYX", 0, 1), ("TYX", 1, 1),
           ("TXY", 0, 0), ("TXY", 1, 0), ("TXY", 0, 1), ("TXY", 1, 1),
           ("YXT", 0, 0), ("YXT", 1, 0), ("YXT", 0, 1), ("YXT", 1, 1),
           ("XYT", 0, 0), ("XYT", 1, 0), ("XYT", 0, 1), ("XYT", 1, 1),
           ("YTX", 0, 0), ("YTX", 1, 0), ("YTX", 0, 1), ("YTX", 1, 1),
           ("XTY", 0, 0), ("XTY", 1, 0), ("XTY", 0, 1), ("XTY", 1, 1))
OBS_KINDS = ((0, 0, 0),)
DONE_SHIFT = False
NORM = True
REMAP = True
MIX = True
HEAD_K = 3
STEM = 64
WIDTH = 40
HPROJ = 32
MEMB = 4
AXES = {"T": 1, "Y": 2, "X": 3}


def lay_in(m, lay, T, Y, X):
    order, fy, fx = lay
    size = {"T": T, "Y": Y, "X": X}
    B = m.shape[0]
    m = m.reshape(B, size[order[0]], size[order[1]], size[order[2]])
    m = m.permute(0, 1 + order.index("T"), 1 + order.index("Y"), 1 + order.index("X"))
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    return m


def lay_out(L, lay):
    order, fy, fx = lay
    if fy:
        L = L.flip(2)
    if fx:
        L = L.flip(3)
    L = L.permute(0, AXES[order[0]], AXES[order[1]], AXES[order[2]])
    return L.reshape(L.shape[0], -1)


def obs_read(o, kind, Y, X):
    k, fy, fx = kind
    B, C = o.shape[0], o.shape[1]
    if k == 1:
        o = o.reshape(B, C, X, Y).transpose(2, 3)
    elif k == 2:
        o = o.reshape(B, Y, X, C).permute(0, 3, 1, 2)
    elif k == 3:
        o = o.reshape(B, X, Y, C).permute(0, 3, 2, 1)
    if fy:
        o = o.flip(2)
    if fx:
        o = o.flip(3)
    return o


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Mix(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        self.tok = nn.Linear(n, n)
        self.ch = nn.Conv2d(3 * c, c, 1)

    def forward(self, f):
        B, C, Y, X = f.shape
        t = F.relu(self.tok(f.reshape(B, C, Y * X))).reshape(B, C, Y, X)
        r = f.mean(3, keepdim=True).expand(B, C, Y, X)
        c = f.mean(2, keepdim=True).expand(B, C, Y, X)
        return F.relu(f + self.ch(torch.cat([t, r, c], 1)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T, Y, X = spec.n_types, spec.Y, spec.X
        C = spec.in_channels
        self.T, self.Y, self.X = T, Y, X
        self.n_place = spec.n_place
        self.n_actions = spec.n_actions
        self.ok = spec.n_place == T * Y * X
        self.layouts = tuple(LAYOUTS) if self.ok else ()
        self.shift = DONE_SHIFT and self.ok and spec.n_actions == spec.n_place + 1
        nsets = 2 if self.shift else 1
        self.nmaps = nsets * len(self.layouts) * T
        self.remap = REMAP
        self.dense = REMAP or (not self.ok)
        in_c = len(OBS_KINDS) * C + self.nmaps + 2
        if self.remap:
            self.obs_remap = nn.Conv1d(C * Y * X, C * Y * X, 1, groups=C)
            in_c += C
        if self.dense:
            self.mask_remap = nn.Linear(spec.n_actions, MEMB * Y * X)
            in_c += MEMB
        stem = [nn.Conv2d(in_c, STEM, 1)]
        if NORM:
            stem.append(nn.GroupNorm(4, STEM))
        stem += [nn.ReLU(), nn.Conv2d(STEM, WIDTH, 1), nn.ReLU()]
        self.stem = nn.Sequential(*stem)
        blocks = []
        for _ in range(2):
            blocks.append(Res(WIDTH))
            if MIX:
                blocks.append(Mix(WIDTH, Y * X))
        self.blocks = nn.ModuleList(blocks)
        self.hconv = nn.Conv2d(WIDTH, HPROJ, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HPROJ, 1)
        if self.nmaps > 0:
            self.out = nn.Conv2d(HPROJ, self.nmaps, 1)
        if self.dense:
            self.oproj = nn.Conv2d(HPROJ, MEMB, 1)
            self.out_remap = nn.Linear(MEMB * Y * X, spec.n_actions)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        if self.shift:
            self.done_head2 = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask.to(obs.dtype)
        feats = [obs_read(obs, k, Y, X) for k in OBS_KINDS]
        if self.remap:
            feats.append(self.obs_remap(obs.reshape(B, -1, 1)).reshape(B, -1, Y, X))
        for lay in self.layouts:
            feats.append(lay_in(m[:, :self.n_place], lay, T, Y, X))
        if self.shift:
            for lay in self.layouts:
                feats.append(lay_in(m[:, 1:self.n_place + 1], lay, T, Y, X))
        if self.dense:
            feats.append(self.mask_remap(m).reshape(B, MEMB, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        f = self.stem(torch.cat(feats, 1))
        for blk in self.blocks:
            f = blk(f)
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        if self.nmaps > 0:
            o = self.out(h)
            n = len(self.layouts)
            place = 0
            for k, lay in enumerate(self.layouts):
                place = place + lay_out(o[:, k * T:(k + 1) * T], lay)
            logits = torch.cat([place, done], 1)
            if self.shift:
                place2 = 0
                for k, lay in enumerate(self.layouts):
                    j = n + k
                    place2 = place2 + lay_out(o[:, j * T:(j + 1) * T], lay)
                logits = logits + torch.cat([self.done_head2(g), place2], 1)
        else:
            logits = torch.zeros(B, self.n_actions, device=obs.device, dtype=obs.dtype)
        if self.dense:
            logits = logits + self.out_remap(self.oproj(h).flatten(1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
