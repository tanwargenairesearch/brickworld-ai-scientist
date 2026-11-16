import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [("TYX", 0, 0), ("TXY", 0, 0), ("TYX", 1, 0), ("TYX", 0, 1)]
OBS8 = True
MIXER = False
CIRC = False
LAYER = False
STEM = 64
WIDTH = 40
HEADC = 32
LW = 16
RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}


def decode(v, lay, T, Y, X):
    order, fy, fx = lay
    dims = {"T": T, "Y": Y, "X": X}
    g = v.reshape(v.shape[0], dims[order[0]], dims[order[1]], dims[order[2]])
    g = g.permute(0, 1 + order.index("T"), 1 + order.index("Y"), 1 + order.index("X"))
    if fy:
        g = g.flip(2)
    if fx:
        g = g.flip(3)
    return g


def encode(m, lay):
    order, fy, fx = lay
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    m = m.permute(0, 1 + "TYX".index(order[0]), 1 + "TYX".index(order[1]), 1 + "TYX".index(order[2]))
    return m.reshape(m.shape[0], -1)


def sym_views(o, sq):
    vs = [o, o.flip(2), o.flip(3), o.flip(2).flip(3)]
    if sq:
        t = o.transpose(2, 3)
        vs = vs + [t, t.flip(2), t.flip(3), t.flip(2).flip(3)]
    return vs


class Mix(nn.Module):
    def __init__(self, c, n):
        super().__init__()
        self.tok = nn.Linear(n, n)
        self.ch = nn.Conv2d(c, c, 1)

    def forward(self, f):
        B, C, Y, X = f.shape
        t = self.tok(f.flatten(2)).reshape(B, C, Y, X)
        return F.relu(f + self.ch(t))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.Z = int(spec.Z)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.ok = (self.n_place == self.T * self.Y * self.X) and (self.n_actions == self.n_place + 1)
        self.sq = self.Y == self.X
        self.nobs = (8 if self.sq else 4) if OBS8 else 1
        self.layer = LAYER and self.C >= 4 * self.Z and self.Z >= 1
        cin = self.C * self.nobs + 2
        if self.ok:
            cin += len(LAYOUTS) * self.T
        else:
            self.mask_embed = nn.Linear(self.n_actions, 4 * self.Y * self.X)
            cin += 4
        ys = torch.linspace(-1, 1, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1, 1, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).clone())
        pm = "circular" if CIRC else "zeros"
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        if self.layer:
            zs = torch.linspace(0, 1, self.Z).view(1, self.Z, 1, 1, 1).expand(1, self.Z, 1, self.Y, self.X)
            self.register_buffer("zpos", zs.clone())
            self.l1 = nn.Conv2d(13, LW, 3, padding=1, padding_mode=pm)
            self.l2 = nn.Conv2d(LW, LW, 1)
            self.lproj = nn.Conv2d(self.Z * LW + LW, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode=pm)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, padding_mode=pm)
        self.hconv = nn.Conv2d(WIDTH, HEADC, 3, padding=1, padding_mode=pm)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        if MIXER:
            n = self.Y * self.X
            self.mixes = nn.ModuleList([Mix(WIDTH, n) for _ in range(3)])
        if self.ok:
            self.outs = nn.Conv2d(HEADC, len(LAYOUTS) * self.T, 1)
        else:
            self.fb_red = nn.Conv2d(HEADC, 8, 1)
            self.fb_out = nn.Linear(8 * self.Y * self.X, self.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def mix(self, f, i):
        if MIXER:
            return self.mixes[i](f)
        return f

    def layer_feats(self, obs):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        parts = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).transpose(1, 2)
        zeros = parts.new_zeros(B, 1, 4, Y, X)
        below = torch.cat([zeros, parts[:, :Z - 1]], 1)
        above = torch.cat([parts[:, 1:], zeros], 1)
        x = torch.cat([parts, below, above, self.zpos.expand(B, -1, -1, -1, -1)], 2)
        x = x.reshape(B * Z, 13, Y, X)
        h = F.relu(self.l1(x))
        h = F.relu(h + self.l2(h))
        h = h.reshape(B, Z, LW, Y, X)
        out = torch.cat([h.reshape(B, Z * LW, Y, X), h.amax(1)], 1)
        return self.lproj(out)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask.float()
        feats = sym_views(obs, self.sq) if OBS8 else [obs]
        if self.ok:
            v = m[:, :self.n_place]
            for lay in LAYOUTS:
                feats.append(decode(v, lay, self.T, self.Y, self.X))
        else:
            feats.append(self.mask_embed(m).reshape(B, 4, self.Y, self.X))
        feats.append(self.coords.expand(B, -1, -1, -1))
        x = torch.cat(feats, 1)
        x = F.relu(self.stem1(x))
        x = self.stem2(x)
        if self.layer:
            x = x + self.layer_feats(obs)
        x = F.relu(x)
        x = self.mix(x, 0)
        f = F.relu(x + self.r1(x))
        f = self.mix(f, 1)
        f = F.relu(f + self.r2(f))
        f = self.mix(f, 2)
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        d = self.done_head(g)
        if self.ok:
            maps = self.outs(h).split(self.T, dim=1)
            p = None
            for k, lay in enumerate(LAYOUTS):
                e = encode(maps[k], lay)
                p = e if p is None else p + e
        else:
            p = self.fb_out(F.relu(self.fb_red(h)).flatten(1))
        logits = torch.cat([p, d], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
