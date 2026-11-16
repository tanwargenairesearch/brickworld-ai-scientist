import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

ORDERS = ("tyx", "txy", "yxt", "xyt")
SHIFTS = (0,)
MIXER = False
OBS_REMAP = True
DENSE_HEAD = False
STEM = 64
W = 40
HP = 32
DR = 8


def flip_dims(fy, fx):
    d = []
    if fy:
        d.append(2)
    if fx:
        d.append(3)
    return d


def to_grid(m, lay, T, Y, X):
    order, fy, fx, s = lay
    size = {"t": T, "y": Y, "x": X}
    g = m.reshape(m.shape[0], size[order[0]], size[order[1]], size[order[2]])
    g = g.permute(0, 1 + order.index("t"), 1 + order.index("y"), 1 + order.index("x"))
    d = flip_dims(fy, fx)
    if d:
        g = g.flip(d)
    return g


def from_grid(g, lay):
    order, fy, fx, s = lay
    d = flip_dims(fy, fx)
    if d:
        g = g.flip(d)
    g = g.permute(0, 1 + "tyx".index(order[0]), 1 + "tyx".index(order[1]), 1 + "tyx".index(order[2]))
    return g.reshape(g.shape[0], -1)


class Block(nn.Module):
    def __init__(self, w, n, mixer):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.mixer = mixer
        if mixer:
            self.tok = nn.Linear(n, n)

    def forward(self, x):
        y = self.conv(x)
        if self.mixer:
            B, C, Y, X = x.shape
            y = y + self.tok(x.reshape(B, C, Y * X)).reshape(B, C, Y, X)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = spec.Y, spec.X, spec.n_types
        self.C = spec.in_channels
        self.n_place = spec.n_place
        self.n_actions = spec.n_actions
        self.n_extra = spec.n_actions - spec.n_place
        self.ok = spec.n_place == self.T * self.Y * self.X
        YX = self.Y * self.X
        shifts = [s for s in SHIFTS if s + self.n_place <= self.n_actions]
        self.lays = []
        if self.ok:
            self.lays = [(o, fy, fx, s) for s in shifts for o in ORDERS for fy in (False, True) for fx in (False, True)]
        self.alt_done = self.ok and any(s > 0 for s in shifts)
        cin = self.C + 2
        if OBS_REMAP:
            self.remap = nn.Linear(YX, YX)
            cin += self.C
        if self.ok:
            cin += self.T * len(self.lays)
        else:
            self.mask_emb = nn.Linear(spec.n_actions, 4 * YX)
            cin += 4
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, W, 1)
        self.blocks = nn.ModuleList([Block(W, YX, MIXER) for _ in range(2)])
        self.hconv = nn.Conv2d(W, HP, 3, padding=1)
        self.hskip = nn.Conv2d(W, HP, 1)
        if self.ok:
            self.outs = nn.Conv2d(HP, self.T * len(self.lays), 1)
        self.use_dense = DENSE_HEAD or not self.ok
        if self.use_dense:
            self.red = nn.Conv2d(HP, DR, 1)
            self.dense = nn.Linear(DR * YX, spec.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, self.n_extra)
        if self.alt_done:
            self.done_alt = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        YX = Y * X
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        ins = [obs, yy, xx]
        if OBS_REMAP:
            ins.append(self.remap(obs.reshape(B, self.C, YX)).reshape(B, self.C, Y, X))
        mf = mask.to(obs.dtype)
        if self.ok:
            for lay in self.lays:
                s = lay[3]
                ins.append(to_grid(mf[:, s:s + self.n_place], lay, T, Y, X))
        else:
            ins.append(self.mask_emb(mf).view(B, 4, Y, X))
        x = F.relu(self.stem2(F.relu(self.stem1(torch.cat(ins, dim=1)))))
        for blk in self.blocks:
            x = blk(x)
        f = x
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        if self.use_dense:
            place0 = self.dense(F.relu(self.red(h)).flatten(1))
        else:
            place0 = torch.zeros(B, self.n_place, device=obs.device, dtype=obs.dtype)
        logits = torch.cat([place0, self.done_head(g)], dim=1)
        if self.ok:
            maps = self.outs(h)
            for k, lay in enumerate(self.lays):
                pk = from_grid(maps[:, k * T:(k + 1) * T], lay)
                s = lay[3]
                logits = logits + F.pad(pk, (s, self.n_actions - self.n_place - s))
        if self.alt_done:
            logits = logits + F.pad(self.done_alt(g), (0, self.n_actions - 1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
