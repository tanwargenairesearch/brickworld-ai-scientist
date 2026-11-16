import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
FEAT = WIDTH + SKIP
GH = 64
P = 16
H = 8
NORD = 2


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
        self.a = nn.Conv2d(n * cin, n * hid, 1, groups=n)
        self.b = nn.Conv2d(n * hid, n, 1, groups=n)

    def forward(self, z):
        return self.b(F.relu(self.a(z)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.Y = spec.Y
        self.X = spec.X
        self.frames = list(range(8)) if spec.X == spec.Y else list(range(4))
        self.G = len(self.frames)
        self.trunk = Trunk(spec.in_channels)
        self.proj = nn.Conv2d(FEAT, P, 1)
        self.heads = GHeads(self.G * NORD * self.T, 3 * P, H)
        self.wh = nn.Parameter(torch.zeros(self.G * NORD))
        self.done_head = nn.Linear(GH, 1)
        self.tbias = nn.Linear(GH, self.T)
        self.value_head = nn.Linear(GH, 1)
        init_all(self)
        ortho(self.heads.b, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.tbias, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        f, g = self.trunk(obs)
        p = self.proj(f)
        feats = []
        for fr in self.frames:
            pg = frame(p, fr)
            pooled = [pool3(pg, dy, dx, (0, dx - 1, 0, dy - 1)) for dy, dx in self.fps]
            for o in range(NORD):
                feats.extend(pooled)
        out = self.heads(torch.cat(feats, dim=1))
        B = out.shape[0]
        out = out.reshape(B, self.G, NORD, self.T, self.Y, self.X)
        out = out + self.tbias(g).view(B, 1, 1, self.T, 1, 1)
        l0 = out[:, :, 0].reshape(B, self.G, -1)
        l1 = out[:, :, 1].permute(0, 1, 3, 4, 2).reshape(B, self.G, -1)
        a = torch.softmax(5.0 * self.wh, dim=0).view(self.G, NORD)
        place = (l0 * a[:, 0].view(1, -1, 1)).sum(1) + (l1 * a[:, 1].view(1, -1, 1)).sum(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
