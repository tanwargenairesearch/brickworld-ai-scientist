import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ent_coef": 0.04, "ppo_epochs": 2}

STEM = 48
COL = 40
WIDTH = 36
PROJ = 24
HID = 32
K = 4


def circ_x(t, X):
    return F.pad(t, (X // 2, X - 1 - X // 2, 0, 0), mode='circular')


def circ_y(t, Y):
    return F.pad(t, (0, 0, Y // 2, Y - 1 - Y // 2), mode='circular')


class ShiftAlign(nn.Module):
    def __init__(self, X, Y, k):
        super().__init__()
        self.X = X
        self.Y = Y
        self.k = k
        self.cx = nn.Conv2d(1, k, (1, X), bias=False)
        self.cy = nn.Conv2d(k, k, (Y, 1), bias=False)

    def forward(self, t):
        B, c, Y, X = t.shape
        u = t.reshape(B * c, 1, Y, X)
        a = self.cx(circ_x(u, self.X))
        a = a + self.cy(circ_y(a, self.Y))
        return a.reshape(B, c * self.k, Y, X)


class AlignStem(nn.Module):
    def __init__(self, cin, X, Y, Z):
        super().__init__()
        self.Z = Z
        self.align = ShiftAlign(X, Y, K)
        n_t = 3 * Z + 1
        self.local = nn.Conv2d(cin + n_t * K, STEM, 1)
        self.out = nn.Conv2d(STEM, COL, 1)

    def forward(self, obs):
        Z = self.Z
        tgt = torch.cat([obs[:, Z:4 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], dim=1)
        al = self.align(tgt)
        h = F.relu(self.local(torch.cat([obs, al], dim=1)))
        return F.relu(self.out(h))


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

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
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = AlignStem(spec.in_channels, spec.X, spec.Y, spec.Z)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
