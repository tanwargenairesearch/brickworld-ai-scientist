import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 1, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

LAYOUTS = ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 1, 1),
           (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 0, 1), (0, 1, 1, 1),
           (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (1, 0, 1, 1),
           (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1), (1, 1, 1, 1))
STEM = 64
WIDTH = 40
HPROJ = 32
LW = 12
LPROJ = 8
LO = 12


def lay_in(m, lay, T, Y, X):
    minor, trans, fy, fx = lay
    B = m.shape[0]
    if trans:
        s1, s2 = X, Y
    else:
        s1, s2 = Y, X
    if minor:
        m = m.reshape(B, s1, s2, T).permute(0, 3, 1, 2)
    else:
        m = m.reshape(B, T, s1, s2)
    if trans:
        m = m.transpose(2, 3)
    if fy:
        m = m.flip(2)
    if fx:
        m = m.flip(3)
    return m


def lay_out(L, lay):
    minor, trans, fy, fx = lay
    if fy:
        L = L.flip(2)
    if fx:
        L = L.flip(3)
    if trans:
        L = L.transpose(2, 3)
    if minor:
        L = L.permute(0, 2, 3, 1)
    return L.reshape(L.shape[0], -1)


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T, Y, X, Z = spec.n_types, spec.Y, spec.X, spec.Z
        self.T, self.Y, self.X, self.Z = T, Y, X, Z
        self.n_place = spec.n_place
        self.n_actions = spec.n_actions
        self.ok = spec.n_place == T * Y * X
        self.layouts = tuple(LAYOUTS) if self.ok else ()
        self.lhead = self.ok and spec.in_channels == 4 * Z + 3
        self.nmaps = len(self.layouts) * T
        in_c = spec.in_channels + self.nmaps + 2
        if not self.ok:
            self.mask_emb = nn.Sequential(nn.Linear(spec.n_actions, 64), nn.Linear(64, 4 * Y * X))
            in_c += 4
        self.stem = nn.Sequential(nn.Conv2d(in_c, STEM, 1), nn.GroupNorm(4, STEM), nn.ReLU(),
                                  nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.res = nn.Sequential(Res(WIDTH), Res(WIDTH))
        self.hconv = nn.Conv2d(WIDTH, HPROJ, 3, padding=1)
        self.hskip = nn.Conv2d(WIDTH, HPROJ, 1)
        if self.lhead:
            self.lproj = nn.Conv2d(WIDTH, LPROJ, 1)
            self.lc1 = nn.Conv2d(9 + Z + LPROJ, LW, 1)
            self.lc2 = nn.Conv2d(LW, LW, 3, padding=1)
            self.lout = nn.Conv2d(LW, LO, 1)
        if self.nmaps > 0:
            self.out = nn.Conv2d(HPROJ + (LO if self.lhead else 0), self.nmaps, 1)
        else:
            self.dproj = nn.Conv2d(HPROJ, 2, 1)
            self.dout = nn.Sequential(nn.Linear(2 * Y * X, 256), nn.ReLU(), nn.Linear(256, spec.n_actions))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def layer_feats(self, obs, f):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        mi = obs[:, 2 * Z:3 * Z]
        ex = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0))[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([b, t, mi, ex, bb, tb], 2)
        hs = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zo = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        pf = self.lproj(f).unsqueeze(1).expand(B, Z, LPROJ, Y, X)
        u = torch.cat([per, hs, zo, pf], 2).reshape(B * Z, -1, Y, X)
        u = F.relu(self.lc1(u))
        u = F.relu(u + self.lc2(u))
        lo = self.lout(u).reshape(B, Z, LO, Y, X)
        return F.relu(lo.amax(1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        m = mask.to(obs.dtype)
        feats = [obs]
        for lay in self.layouts:
            feats.append(lay_in(m[:, :self.n_place], lay, T, Y, X))
        if not self.ok:
            feats.append(self.mask_emb(m).reshape(B, 4, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        x = torch.cat(feats, 1)
        f = self.res(self.stem(x))
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        done = self.done_head(g)
        if self.nmaps > 0:
            if self.lhead:
                h = torch.cat([h, self.layer_feats(obs, f)], 1)
            o = self.out(h)
            place = 0
            for k, lay in enumerate(self.layouts):
                place = place + lay_out(o[:, k * T:(k + 1) * T], lay)
            logits = torch.cat([place, done], 1)
        else:
            logits = self.dout(self.dproj(h).flatten(1))
        value = self.value_head(g).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
