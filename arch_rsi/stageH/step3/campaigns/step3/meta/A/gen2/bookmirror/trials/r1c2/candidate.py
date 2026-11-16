import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
SQ = 32
GH = 64
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


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
        self.T = spec.n_types
        self.K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.frames = [0, 4] if spec.X == spec.Y else [0, 3]
        nf = len(self.frames)
        self.nf = nf
        self.convs = nn.ModuleList([ortho(nn.Conv2d(self.base.ch, SQ, self.K), math.sqrt(2)) for _ in self.frames])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(SQ, self.T, 1), 0.01) for _ in self.frames])
        self.tmix = nn.Parameter(torch.zeros(nf, self.T, self.T))
        self.dense = ortho(nn.Linear(GH, spec.n_actions), 0.01)
        self.alpha = nn.Parameter(torch.zeros(nf))

    def forward(self, obs, mask):
        f, g = self.base.features(obs)
        B = f.shape[0]
        K = self.K
        eye = torch.eye(self.T, device=f.device, dtype=f.dtype)
        maps = []
        for i, k in enumerate(self.frames):
            u = dihedral(f, k)
            u = F.pad(u, (0, K - 1, 0, K - 1))
            h = F.relu(self.convs[i](u))
            m = self.outs[i](h)
            M = eye + self.tmix[i]
            m = torch.einsum('st,btyx->bsyx', M, m)
            maps.append(m.reshape(B, -1))
        L = torch.stack(maps, 1)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = (L * w.view(1, self.nf, 1)).sum(1)
        n = place.shape[1] // self.T
        place = place + self.base.tbias(g).unsqueeze(2).expand(B, self.T, n).reshape(B, -1)
        logits = torch.cat([place, self.base.done(g)], 1) + self.dense(g)
        return logits, self.base.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
