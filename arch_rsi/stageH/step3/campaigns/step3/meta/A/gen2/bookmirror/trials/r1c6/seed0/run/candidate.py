import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 16
HID = 16
GH = 64
SCALE = 5.0
CORNERS = ((True, True), (True, False), (False, True), (False, False))


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class GLin(nn.Module):
    def __init__(self, G, cin, cout, gain):
        super().__init__()
        w = torch.empty(G, cin, cout)
        for g in range(G):
            nn.init.orthogonal_(w[g], gain)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(G, cout))
        self.G = G
        self.cout = cout

    def forward(self, x):
        y = torch.einsum('bgcyx,gco->bgoyx', x, self.w)
        return y + self.b.view(1, self.G, self.cout, 1, 1)


def dihedral(x, k):
    if k >= 4:
        x = x.transpose(-2, -1)
    m = k % 4
    if m == 1:
        x = x.flip(-1)
    elif m == 2:
        x = x.flip(-2)
    elif m == 3:
        x = x.flip(-2).flip(-1)
    return x


def wpool(q, dy, dx, top=True, left=True):
    if dy * dx == 1:
        return q, q, q
    pad = ((0, dx - 1) if left else (dx - 1, 0)) + ((0, dy - 1) if top else (dy - 1, 0))
    q = F.pad(q, pad)
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Base(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.s1 = ortho(nn.Conv2d(c, STEM, 1), math.sqrt(2))
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), math.sqrt(2))
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), math.sqrt(2))
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), math.sqrt(2))
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), 1.0)
        self.ch = WIDTH + SKIP
        self.glob = nn.Sequential(ortho(nn.Linear(2 * self.ch, GH), math.sqrt(2)), nn.ReLU(),
                                  ortho(nn.Linear(GH, GH), math.sqrt(2)), nn.ReLU())
        self.done = ortho(nn.Linear(GH, 1), 0.01)
        self.value = ortho(nn.Linear(GH, 1), 1.0)
        self.tbias = ortho(nn.Linear(GH, spec.n_types), 0.01)

    def features(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, self.skip(obs)], 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        return f, g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.base = Base(spec)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        sq = spec.X == spec.Y
        self.frames = list(range(8)) if sq else [0, 1, 2, 3]
        self.no = 2 if sq else 1
        G = len(CORNERS) * self.no
        self.G = G
        self.proj = ortho(nn.Conv2d(self.base.ch, PROJ, 1), math.sqrt(2))
        self.mix = nn.ModuleList([GLin(G, 3 * PROJ, HID, math.sqrt(2)) for _ in self.fps])
        self.out = nn.ModuleList([GLin(G, HID, 1, 0.01) for _ in self.fps])
        self.af = nn.Parameter(torch.zeros(len(self.frames)))
        self.ac = nn.Parameter(torch.zeros(len(CORNERS)))
        self.ao = nn.Parameter(torch.zeros(2))

    def forward(self, obs, mask):
        f, g = self.base.features(obs)
        p = self.proj(f)
        B, _, Y, X = p.shape
        nc = len(CORNERS)
        per_t = []
        for t, (dy, dx) in enumerate(self.fps):
            zs = []
            for top, left in CORNERS:
                for o in range(self.no):
                    a, b = (dy, dx) if o == 0 else (dx, dy)
                    mx, mn, av = wpool(p, a, b, top, left)
                    zs.append(torch.cat([mx, mn, av], 1))
            z = torch.stack(zs, 1)
            h = F.relu(self.mix[t](z))
            per_t.append(self.out[t](h))
        O = torch.cat(per_t, 2).reshape(B, nc, self.no, self.T, Y, X)
        wc = torch.softmax(SCALE * self.ac, 0)
        wf = torch.softmax(SCALE * self.af, 0)
        wo = torch.softmax(SCALE * self.ao, 0)
        Oc = (O * wc.view(1, nc, 1, 1, 1, 1)).sum(1)
        tb = self.base.tbias(g).view(B, self.T, 1, 1)
        place = None
        for i, k in enumerate(self.frames):
            o = 1 if k >= 4 else 0
            A = dihedral(Oc[:, o] + tb, k)
            tm = A.reshape(B, -1)
            am = A.permute(0, 2, 3, 1).reshape(B, -1)
            term = wf[i] * (wo[0] * tm + wo[1] * am)
            place = term if place is None else place + term
        logits = torch.cat([place, self.base.done(g)], 1)
        return logits, self.base.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
