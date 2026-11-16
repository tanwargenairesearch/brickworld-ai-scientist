import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'ppo_epochs': 2, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 16, 'gamma': 0.97, 'gae_lambda': 0.95, 'ent_coef': 0.005}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 12
HID = 8
GH = 64


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def tf(x, k):
    if k & 4:
        x = x.transpose(2, 3)
    if k & 1:
        x = x.flip(2)
    if k & 2:
        x = x.flip(3)
    return x


def pool3(q, dy, dx, cy=0, cx=0):
    if dy == 1 and dx == 1:
        return torch.cat([q, q, q], 1)
    pt = cy * (dy - 1)
    pb = dy - 1 - pt
    pl = cx * (dx - 1)
    pr = dx - 1 - pl
    z = F.pad(q, (pl, pr, pt, pb))
    mx = F.max_pool2d(z, (dy, dx), stride=1)
    mn = -F.max_pool2d(-z, (dy, dx), stride=1)
    av = F.avg_pool2d(z, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], 1)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        r2 = math.sqrt(2.0)
        self.s1 = init_(nn.Conv2d(c, STEM, 1), r2)
        self.s2 = init_(nn.Conv2d(STEM, COL, 1), r2)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), r2)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), r2)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = init_(nn.Conv2d(c, SKIP, 1), r2)
        self.glob = nn.Sequential(init_(nn.Linear(2 * WIDTH, GH), r2), nn.ReLU())
        self.done = init_(nn.Linear(GH, 1), 0.01)
        self.value = init_(nn.Linear(GH, 1), 1.0)
        self.tbias = init_(nn.Linear(GH, spec.n_types), 0.01)

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        u = torch.cat([f, F.relu(self.skip(obs))], 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        return u, g


class TypeHeads(nn.Module):
    def __init__(self, fps, n_out=1, hid=HID, cy=0, cx=0):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in fps]
        self.n_out = n_out
        self.cy = cy
        self.cx = cx
        r2 = math.sqrt(2.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, n_out * hid, 1), r2) for _ in self.fps])
        self.out = nn.ModuleList([init_(nn.Conv2d(n_out * hid, n_out, 1), 0.01) for _ in self.fps])

    def forward(self, q):
        res = []
        for (dy, dx), mx, ot in zip(self.fps, self.mix, self.out):
            res.append(ot(F.relu(mx(pool3(q, dy, dx, self.cy, self.cx)))))
        return torch.stack(res, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.proj = init_(nn.Conv2d(WIDTH + SKIP, PROJ, 1), math.sqrt(2.0))
        if int(spec.X) == int(spec.Y):
            self.frames = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            self.frames = [0, 1, 2, 3]
        self.fheads = nn.ModuleList([TypeHeads(spec.footprints, 2, HID) for _ in self.frames])
        self.corners = [(0, 1), (1, 0), (1, 1)]
        self.cheads = nn.ModuleList([TypeHeads(spec.footprints, 1, HID, cy, cx) for cy, cx in self.corners])
        self.hyp = nn.Parameter(torch.zeros(2 * len(self.frames) + len(self.corners)))

    def forward(self, obs, mask):
        u, g = self.trunk(obs)
        p = self.proj(u)
        tb = self.trunk.tbias(g)[:, :, None, None]
        w = F.softmax(5.0 * self.hyp, 0)
        flats = []
        for i, k in enumerate(self.frames):
            o = self.fheads[i](tf(p, k))
            m0 = o[:, :, 0] + tb
            m1 = o[:, :, 1] + tb
            flats.append(m0.flatten(1))
            flats.append(m1.permute(0, 2, 3, 1).flatten(1))
        for h in self.cheads:
            flats.append((h(p)[:, :, 0] + tb).flatten(1))
        place = None
        for j, fl in enumerate(flats):
            term = w[j] * fl
            place = term if place is None else place + term
        logits = torch.cat([place, self.trunk.done(g)], 1)
        return logits, self.trunk.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
