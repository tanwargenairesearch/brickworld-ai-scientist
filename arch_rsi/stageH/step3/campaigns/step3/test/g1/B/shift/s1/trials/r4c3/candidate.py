import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

POOL = True
CORNERS = [(0, 0)]
ZSTRUCT = True
STEM = 64
WIDTH = 40
HEAD = 32
GH = 64
PROJ = 24
PHID = 24
ZW = 32
EMB = 8


def make_layouts():
    out = []
    for o in (0, 1):
        for tr in (0, 1):
            for fy in (0, 1):
                for fx in (0, 1):
                    out.append((o, tr, fy, fx))
    return out


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


class PoolHead(nn.Module):
    def __init__(self, c, fps, proj, hid, corners):
        super().__init__()
        self.fps = fps
        self.corners = list(corners)
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj * len(self.corners), hid, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            feats = []
            for ay, ax in self.corners:
                if dy * dx == 1:
                    feats += [p, p, p]
                else:
                    pad = (dx - 1 if ax else 0, 0 if ax else dx - 1, dy - 1 if ay else 0, 0 if ay else dy - 1)
                    q = F.pad(p, pad)
                    mx = F.max_pool2d(q, (dy, dx), stride=1)
                    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                    av = F.avg_pool2d(q, (dy, dx), stride=1)
                    feats += [mx, mn, av]
            h = F.relu(mix(torch.cat(feats, 1)))
            res.append(out(h))
        return torch.cat(res, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y, self.X, self.Z = int(spec.Y), int(spec.X), int(spec.Z)
        self.T = int(spec.n_types)
        self.C = int(spec.in_channels)
        self.n_place = int(spec.n_place)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Y, X, Z, T, C = self.Y, self.X, self.Z, self.T, self.C
        YX = Y * X
        self.ok = self.n_place == T * YX
        self.lays = make_layouts() if self.ok else []
        self.use_pool = POOL and self.ok and len(self.fps) == T and all(dy <= Y and dx <= X for dy, dx in self.fps)
        self.zs = ZSTRUCT and C == 4 * Z + 3
        cin = C + 2 + T * len(self.lays)
        if not self.ok:
            self.memb = nn.Linear(self.n_place, EMB * YX)
            cin += EMB
        self.s1 = nn.Conv2d(cin, STEM, 1)
        self.s2 = nn.Conv2d(STEM, WIDTH, 1)
        if self.zs:
            self.z1a = nn.Conv2d(7, ZW, 1)
            self.z1b = nn.Conv2d(3, ZW, 1)
            self.z2 = nn.Conv2d(ZW, ZW, 1)
            self.zp = nn.Conv2d(2 * ZW, WIDTH, 1)
        self.r1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.r2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.hk = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
        self.hs = nn.Conv2d(WIDTH, HEAD, 1)
        if self.ok:
            self.out = nn.Conv2d(HEAD, T * len(self.lays), 1)
        else:
            self.fo = nn.Conv2d(HEAD, 4, 1)
            self.fl = nn.Linear(4 * YX, self.n_place)
        if self.use_pool:
            self.pool = PoolHead(WIDTH, self.fps, PROJ, PHID, CORNERS)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done = nn.Linear(GH, 1)
        self.value = nn.Linear(GH, 1)

    def zfeat(self, obs):
        B = obs.shape[0]
        Y, X, Z = self.Y, self.X, self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        mi = obs[:, 2 * Z:3 * Z]
        ex = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        zp = torch.linspace(-1.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1).expand(B, Z, Y, X)
        s = torch.stack([b, t, mi, ex, bb, tb, zp], 2).reshape(B * Z, 7, Y, X)
        z = self.z1a(s).view(B, Z, ZW, Y, X) + self.z1b(obs[:, 4 * Z:4 * Z + 3]).unsqueeze(1)
        z = F.relu(z).reshape(B * Z, ZW, Y, X)
        z = F.relu(self.z2(z)).view(B, Z, ZW, Y, X)
        return torch.cat([z.amax(1), z.mean(1)], 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        mp = mask.float()[:, :self.n_place]
        feats = [obs]
        for lay in self.lays:
            feats.append(to_grid(mp, lay, T, Y, X))
        if not self.ok:
            feats.append(self.memb(mp).view(B, EMB, Y, X))
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        feats.append(yy)
        feats.append(xx)
        x = self.s2(F.relu(self.s1(torch.cat(feats, 1))))
        if self.zs:
            x = x + self.zp(self.zfeat(obs))
        x = F.relu(x)
        x = F.relu(x + self.r1(x))
        x = F.relu(x + self.r2(x))
        h = F.relu(self.hk(x) + self.hs(x))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        if self.ok:
            o = self.out(h)
            place = 0
            for i, lay in enumerate(self.lays):
                place = place + from_grid(o[:, i * T:(i + 1) * T], lay)
            if self.use_pool:
                place = place + self.pool(x).flatten(1)
        else:
            place = self.fl(F.relu(self.fo(h)).flatten(1))
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
