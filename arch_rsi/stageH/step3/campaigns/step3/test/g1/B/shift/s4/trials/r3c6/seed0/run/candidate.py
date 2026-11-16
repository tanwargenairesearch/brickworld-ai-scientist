import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

ORDERS = ("tyx", "txy", "ytx", "xty", "yxt", "xyt")
ADAPT = True
SEP_CRITIC = True
CRIT_SCALE = 0.1
FLATMASK = True
LAYERS = True
STEM = 64
W = 40
HP = 32
LW = 16
VW = 32


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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T, self.Z = spec.Y, spec.X, spec.n_types, spec.Z
        self.C = spec.in_channels
        self.n_place = spec.n_place
        self.n_extra = spec.n_actions - spec.n_place
        self.ok = spec.n_place == self.T * self.Y * self.X
        self.lays = make_layouts()
        oc = self.C
        if ADAPT:
            self.adapt = nn.Conv2d(self.C, 2 * self.C, 1, groups=self.C)
            oc = 2 * self.C
        cin = oc + 2
        if self.ok:
            cin += self.T * len(self.lays)
        else:
            self.mask_emb = nn.Linear(spec.n_actions, 4 * self.Y * self.X)
            cin += 4
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, W, 1)
        self.use_layers = LAYERS and self.C == 4 * self.Z + 3 and self.Z >= 2
        if self.use_layers:
            self.l1 = nn.Conv2d(14, LW, 1)
            self.l2 = nn.Conv2d(LW, LW, 3, padding=1)
            self.l3 = nn.Conv2d(2 * LW, W, 1)
        self.r1 = nn.Conv2d(W, W, 3, padding=1)
        self.r2 = nn.Conv2d(W, W, 3, padding=1)
        self.hconv = nn.Conv2d(W, HP, 3, padding=1)
        self.hskip = nn.Conv2d(W, HP, 1)
        if self.ok:
            self.outs = nn.Conv2d(HP, self.T * len(self.lays), 1)
        else:
            self.red = nn.Conv2d(HP, 8, 1)
            self.dense = nn.Linear(8 * self.Y * self.X, spec.n_place)
        gin = 2 * W
        if FLATMASK:
            self.mflat = nn.Linear(spec.n_actions, 64)
            gin += 64
        self.glob = nn.Sequential(nn.Linear(gin, 64), nn.ReLU())
        self.done_head = nn.Linear(64, self.n_extra)
        if FLATMASK:
            self.gbias = nn.Linear(64, spec.n_actions)
        if SEP_CRITIC:
            self.vstem = nn.Conv2d(cin, VW, 1)
            self.vnet = nn.Sequential(nn.Linear(2 * W + 2 * VW, 64), nn.ReLU(), nn.Linear(64, 1))
        else:
            self.value_head = nn.Linear(64, 1)

    def layer_branch(self, obs, yy, xx):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        ob = obs[:, 0:Z]
        tg = obs[:, Z:2 * Z]
        ms = obs[:, 2 * Z:3 * Z]
        ex = obs[:, 3 * Z:4 * Z]
        obd = F.pad(ob[:, :Z - 1], (0, 0, 0, 0, 1, 0))
        tgd = F.pad(tg[:, :Z - 1], (0, 0, 0, 0, 1, 0))
        obu = F.pad(ob[:, 1:], (0, 0, 0, 0, 0, 1))
        tgu = F.pad(tg[:, 1:], (0, 0, 0, 0, 0, 1))
        zp = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1).expand(B, Z, Y, X)
        per = torch.stack([ob, tg, ms, ex, obd, tgd, obu, tgu, zp], dim=2)
        sh = torch.cat([obs[:, 4 * Z:4 * Z + 3], yy, xx], dim=1).unsqueeze(1).expand(B, Z, 5, Y, X)
        t = torch.cat([per, sh], dim=2).reshape(B * Z, 14, Y, X)
        h = F.relu(self.l1(t))
        h = F.relu(h + self.l2(h))
        h = h.reshape(B, Z, LW, Y, X)
        agg = torch.cat([h.amax(1), h.mean(1)], dim=1)
        return self.l3(agg)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = obs
        if ADAPT:
            a = self.adapt(obs)
            o = a / (1.0 + a.abs())
        ins = [o, yy, xx]
        mf = mask.to(obs.dtype)
        if self.ok:
            mp = mf[:, :self.n_place]
            for lay in self.lays:
                ins.append(to_grid(mp, lay, T, Y, X))
        else:
            ins.append(self.mask_emb(mf).view(B, 4, Y, X))
        inp = torch.cat(ins, dim=1)
        x = F.relu(self.stem2(F.relu(self.stem1(inp))))
        if self.use_layers:
            x = F.relu(x + self.layer_branch(obs, yy, xx))
        x = F.relu(x + self.r1(x))
        f = F.relu(x + self.r2(x))
        h = F.relu(self.hconv(f) + self.hskip(f))
        if self.ok:
            maps = self.outs(h)
            place = None
            for k, lay in enumerate(self.lays):
                pk = from_grid(maps[:, k * T:(k + 1) * T], lay)
                place = pk if place is None else place + pk
        else:
            place = self.dense(F.relu(self.red(h)).flatten(1))
        fm = f.mean((2, 3))
        fx = f.amax((2, 3))
        gl = [fm, fx]
        if FLATMASK:
            gl.append(F.relu(self.mflat(mf)))
        g = self.glob(torch.cat(gl, dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        if FLATMASK:
            logits = logits + self.gbias(g)
        if SEP_CRITIC:
            vs = F.relu(self.vstem(inp))
            vf = torch.cat([fm.detach(), fx.detach(), vs.mean((2, 3)), vs.amax((2, 3))], dim=1)
            v = self.vnet(vf).squeeze(1)
            v = CRIT_SCALE * v + (1.0 - CRIT_SCALE) * v.detach()
        else:
            v = self.value_head(g).squeeze(1)
        return logits, v


def build_policy(spec):
    return Net(spec)
