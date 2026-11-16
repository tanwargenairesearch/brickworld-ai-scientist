import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

WIDTH = 40
SKIP = 16
PROJ = 24
HID = 24
GHID = 64
SCALE = 2.0


def frame_list(Y, X):
    out = []
    for tr in ([0, 1] if Y == X else [0]):
        for fy in (0, 1):
            for fx in (0, 1):
                out.append((tr, fy, fx))
    return out


def to_orig(tr, fy, fx, Yf, Xf, yy, xx):
    a = Yf - 1 - yy if fy else yy
    b = Xf - 1 - xx if fx else xx
    if tr:
        return b, a
    return a, b


def init_net(m):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(mod.weight, math.sqrt(2))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.stem = nn.Sequential(nn.Conv2d(C, 64, 1), nn.ReLU(), nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(C, SKIP, 1)
        self.out_ch = WIDTH + SKIP
        self.glob = nn.Sequential(nn.Linear(2 * self.out_ch, GHID), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, self.skip(obs)], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return f, g


class FramePoolHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        T = int(spec.n_types)
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        frames = frame_list(Y, X)
        nf = len(frames)
        shapes = []
        pair_shape = []
        idx_rows = []
        for (tr, fy, fx) in frames:
            Yf, Xf = (X, Y) if tr else (Y, X)
            for (dy, dx) in fps:
                s = (dx, dy) if tr else (dy, dx)
                if s not in shapes:
                    shapes.append(s)
                pair_shape.append(shapes.index(s))
                row = []
                for yy in range(Yf):
                    for xx in range(Xf):
                        if yy + dy > Yf or xx + dx > Xf:
                            row.append(Y * X)
                            continue
                        ys = []
                        xs = []
                        for i in range(dy):
                            for j in range(dx):
                                oy, ox = to_orig(tr, fy, fx, Yf, Xf, yy + i, xx + j)
                                ys.append(oy)
                                xs.append(ox)
                        row.append(min(ys) * X + min(xs))
                idx_rows.append(row)
        nP = nf * T
        N = Y * X + 1
        flat = []
        for k in range(nP):
            base = (pair_shape[k] * nP + k) * N
            flat.extend([base + v for v in idx_rows[k]])
        self.register_buffer("gidx", torch.tensor(flat, dtype=torch.long))
        self.shapes = shapes
        self.nf = nf
        self.T = T
        self.YX = Y * X
        self.proj = nn.Conv2d(cin, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in shapes])
        self.read = nn.ModuleList([nn.Conv2d(HID, nP, 1) for _ in shapes])

    def forward(self, f):
        B = f.shape[0]
        p = self.proj(f)
        outs = []
        for (sy, sx), mix, rd in zip(self.shapes, self.mix, self.read):
            if sy * sx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, sx - 1, 0, sy - 1))
                mx = F.max_pool2d(q, (sy, sx), stride=1)
                mn = -F.max_pool2d(-q, (sy, sx), stride=1)
                av = F.avg_pool2d(q, (sy, sx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            outs.append(rd(h).flatten(2))
        O = torch.stack(outs, 1)
        O = F.pad(O, (0, 1))
        L = O.flatten(1)[:, self.gidx]
        return L.view(B, self.nf, self.T, self.YX)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.trunk = Trunk(spec)
        self.head = FramePoolHead(spec, self.trunk.out_ch)
        self.mixw = nn.Parameter(torch.zeros(self.T, self.head.nf))
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.tbias = nn.Linear(GHID, self.T)
        init_net(self)
        for m in [self.done_head, self.tbias] + list(self.head.read):
            nn.init.orthogonal_(m.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        f, g = self.trunk(obs)
        L = self.head(f)
        w = torch.softmax(SCALE * self.mixw, 1)
        place = torch.einsum('tf,bftn->btn', w, L)
        place = place + self.tbias(g).unsqueeze(2)
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
