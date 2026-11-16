import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUTS = [("TYX", 0, 0), ("TXY", 0, 0), ("TYX", 1, 0), ("TYX", 0, 1)]
OBS8 = False
MIXER = False
NORM = True
CRITIC = True
HEAD_K = 3
STEM = 64
WIDTH = 40
HEADC = 32
CW = 24
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


class Critic(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.c1 = nn.Conv2d(cin, CW, 1)
        self.n1 = nn.GroupNorm(4, CW)
        self.c2 = nn.Conv2d(CW, CW, 3, padding=1)
        self.fc1 = nn.Linear(2 * CW, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = F.relu(h + self.c2(h))
        z = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.fc2(F.relu(self.fc1(z))).squeeze(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.T = int(spec.Y), int(spec.X), int(spec.n_types)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.ok = (self.n_place == self.T * self.Y * self.X) and (self.n_actions == self.n_place + 1)
        self.sq = self.Y == self.X
        self.nobs = (8 if self.sq else 4) if OBS8 else 1
        cin = self.C * self.nobs + 2
        if self.ok:
            cin += len(LAYOUTS) * self.T
        else:
            self.mask_embed = nn.Linear(self.n_actions, 4 * self.Y * self.X)
            cin += 4
        ys = torch.linspace(-1, 1, self.Y).view(1, 1, self.Y, 1).expand(1, 1, self.Y, self.X)
        xs = torch.linspace(-1, 1, self.X).view(1, 1, 1, self.X).expand(1, 1, self.Y, self.X)
        self.register_buffer("coords", torch.cat([ys, xs], 1).clone())
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        if NORM:
            self.n1 = nn.GroupNorm(8, STEM)
            self.n2 = nn.GroupNorm(8, WIDTH)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hconv = nn.Conv2d(WIDTH, HEADC, HEAD_K, padding=HEAD_K // 2)
        self.hskip = nn.Conv2d(WIDTH, HEADC, 1)
        if MIXER:
            self.mixes = nn.ModuleList([Mix(WIDTH, self.Y * self.X) for _ in range(3)])
        if self.ok:
            self.outs = nn.Conv2d(HEADC, len(LAYOUTS) * self.T, 1)
        else:
            self.fb_red = nn.Conv2d(HEADC, 8, 1)
            self.fb_out = nn.Linear(8 * self.Y * self.X, self.n_actions - 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        if CRITIC:
            self.critic = Critic(self.C + 2)
        else:
            self.value_head = nn.Linear(64, 1)

    def mix(self, f, i):
        if MIXER:
            return self.mixes[i](f)
        return f

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask.float()
        crd = self.coords.expand(B, -1, -1, -1)
        feats = sym_views(obs, self.sq) if OBS8 else [obs]
        if self.ok:
            v = m[:, :self.n_place]
            for lay in LAYOUTS:
                feats.append(decode(v, lay, self.T, self.Y, self.X))
        else:
            feats.append(self.mask_embed(m).reshape(B, 4, self.Y, self.X))
        feats.append(crd)
        x = self.stem1(torch.cat(feats, 1))
        if NORM:
            x = self.n1(x)
        x = self.stem2(F.relu(x))
        if NORM:
            x = self.n2(x)
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
        if CRITIC:
            value = self.critic(torch.cat([obs, crd], 1))
        else:
            value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
