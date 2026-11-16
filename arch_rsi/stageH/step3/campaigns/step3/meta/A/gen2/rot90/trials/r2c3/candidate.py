import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_minibatches": 64}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
FEAT = 64
GH = 64
P = 16
H = 16


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def init_all(net):
    for mod in net.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            ortho(mod, math.sqrt(2))


def frame(t, k):
    if k >= 4:
        t = t.transpose(2, 3)
    if k % 2 == 1:
        t = t.flip(3)
    if (k // 2) % 2 == 1:
        t = t.flip(2)
    return t


def pool3(p, dy, dx, pad):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, pad)
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * FEAT, GH), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, self.skip(obs)], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        return f, g


class GHeads(nn.Module):
    def __init__(self, n, cin, hid):
        super().__init__()
        self.wa = nn.Parameter(torch.empty(n, cin, hid))
        self.ba = nn.Parameter(torch.zeros(n, 1, hid))
        self.wb = nn.Parameter(torch.empty(n, hid, 1))
        self.bb = nn.Parameter(torch.zeros(n, 1, 1))
        for i in range(n):
            nn.init.orthogonal_(self.wa.data[i], math.sqrt(2))
            nn.init.orthogonal_(self.wb.data[i], 0.01)

    def forward(self, feats):
        z = torch.stack(feats, 0)
        n, B, C, Y, X = z.shape
        z = z.permute(0, 1, 3, 4, 2).reshape(n, B * Y * X, C)
        h = F.relu(torch.bmm(z, self.wa) + self.ba)
        o = torch.bmm(h, self.wb) + self.bb
        return o.reshape(n, B, Y, X).permute(1, 0, 2, 3)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.frames = list(range(8)) if spec.X == spec.Y else list(range(4))
        self.G = len(self.frames)
        self.trunk = Trunk(spec.in_channels)
        self.proj = nn.Conv2d(FEAT, P, 1)
        self.done_head = nn.Linear(GH, 1)
        self.tbias = nn.Linear(GH, self.T)
        self.value_head = nn.Linear(GH, 1)
        init_all(self)
        ortho(self.done_head, 0.01)
        ortho(self.tbias, 0.01)
        ortho(self.value_head, 1.0)
        self.heads = GHeads(self.G * self.T, 3 * P, H)
        self.wf = nn.Parameter(torch.zeros(self.G))
        self.wperm = nn.Parameter(torch.zeros(self.T, self.T))
        self.wsel = nn.Parameter(torch.zeros(2))

    def forward(self, obs, mask):
        f, g = self.trunk(obs)
        p = self.proj(f)
        feats = []
        for fr in self.frames:
            pg = frame(p, fr)
            for dy, dx in self.fps:
                feats.append(pool3(pg, dy, dx, (0, dx - 1, 0, dy - 1)))
        out = self.heads(feats)
        B = out.shape[0]
        out = out.reshape(B, self.G, self.T, -1)
        a = torch.softmax(5.0 * self.wf, dim=0)
        hs = (out * a.view(1, -1, 1, 1)).sum(1)
        A = torch.softmax(5.0 * self.wperm, dim=1)
        lperm = torch.einsum('ts,bsn->btn', A, hs)
        c = torch.softmax(5.0 * self.wsel, dim=0)
        place = c[0] * hs + c[1] * lperm + self.tbias(g).unsqueeze(2)
        logits = torch.cat([place.reshape(B, -1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
