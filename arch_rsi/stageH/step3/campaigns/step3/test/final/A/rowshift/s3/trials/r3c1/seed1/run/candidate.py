import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
PROJ = 32
HID = 32
GLOB = 64
CRIT = 24
SCALE = 5.0
VGRAD = 0.1


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def frame_maps(Y, X):
    maps = [lambda y, x: (y, x),
            lambda y, x: (y, X - 1 - x),
            lambda y, x: (Y - 1 - y, x),
            lambda y, x: (Y - 1 - y, X - 1 - x)]
    if Y == X:
        maps = maps + [lambda y, x: (x, y),
                       lambda y, x: (x, X - 1 - y),
                       lambda y, x: (Y - 1 - x, y),
                       lambda y, x: (Y - 1 - x, X - 1 - y)]
    return maps


def build_index(Y, X, fps, maps):
    T = len(fps)
    P = T * Y * X
    nf = len(maps)
    lookup = {}
    for i, fp in enumerate(fps):
        if fp not in lookup:
            lookup[fp] = i
    rows = []
    for g, m in enumerate(maps):
        row = []
        for t in range(T):
            dy, dx = fps[t]
            for y in range(Y):
                for x in range(X):
                    if y + dy > Y or x + dx > X:
                        row.append(nf * P)
                        continue
                    cells = [m(y + i, x + j) for i in range(dy) for j in range(dx)]
                    ys = [c[0] for c in cells]
                    xs = [c[1] for c in cells]
                    y0 = min(ys)
                    x0 = min(xs)
                    shape = (max(ys) - y0 + 1, max(xs) - x0 + 1)
                    if shape in lookup:
                        s = lookup[shape]
                        row.append(g * P + s * Y * X + y0 * X + x0)
                    else:
                        row.append(nf * P)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


class FrameHead(nn.Module):
    def __init__(self, c, fps, nf):
        super().__init__()
        self.fps = fps
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, nf, 1) for _ in fps])

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
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        return torch.stack(res, dim=2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        C = int(spec.in_channels)
        self.Y = Y
        self.X = X
        self.C = C
        cin = 3 * C
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(fps)
        maps = frame_maps(Y, X)
        self.nf = len(maps)
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.head = FrameHead(WIDTH + SKIP, fps, self.nf)
        self.tbias = nn.Linear(GLOB, T)
        self.done_head = nn.Linear(GLOB, 1)
        self.cconv = nn.Conv2d(cin, CRIT, 1)
        self.cmlp = nn.Linear(2 * CRIT + GLOB, GLOB)
        self.value_head = nn.Linear(GLOB, 1)
        self.frame_logit = nn.Parameter(torch.zeros(self.nf))
        self.register_buffer("idx", build_index(Y, X, fps, maps))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for o in self.head.outs:
            ortho(o, 0.01)
        ortho(self.tbias, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def views(self, obs):
        B = obs.shape[0]
        o2 = obs.reshape(B, self.Y, self.X, self.C).permute(0, 3, 1, 2)
        o3 = obs.reshape(B, self.X, self.Y, self.C).permute(0, 3, 2, 1)
        return torch.cat([obs, o2, o3], dim=1)

    def forward(self, obs, mask):
        v = self.views(obs)
        s = self.stem(v)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        h = torch.cat([f, self.skip(v)], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        sc = self.head(h)
        sc = sc + self.tbias(g)[:, None, :, None, None]
        flat = sc.flatten(1)
        flat = torch.cat([flat, flat.new_zeros(flat.shape[0], 1)], dim=1)
        per = flat[:, self.idx]
        w = torch.softmax(SCALE * self.frame_logit, 0)
        place = torch.einsum('bfp,f->bp', per, w)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        c = F.relu(self.cconv(v))
        cg = torch.cat([c.mean((2, 3)), c.amax((2, 3)), g.detach()], dim=1)
        val = self.value_head(F.relu(self.cmlp(cg))).squeeze(1)
        val = val.detach() + VGRAD * (val - val.detach())
        return logits, val


def build_policy(spec):
    return Net(spec)
