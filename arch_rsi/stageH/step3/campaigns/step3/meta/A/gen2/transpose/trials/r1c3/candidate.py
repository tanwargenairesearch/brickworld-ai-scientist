import math
import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 16
HID = 8
GH = 64
NORD = 6


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


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


def order_flat(m, o):
    if o == 0:
        return m.flatten(1)
    if o == 1:
        return m.transpose(2, 3).flatten(1)
    if o == 2:
        return m.permute(0, 2, 3, 1).flatten(1)
    if o == 3:
        return m.permute(0, 3, 2, 1).flatten(1)
    if o == 4:
        return m.permute(0, 2, 1, 3).flatten(1)
    return m.permute(0, 3, 1, 2).flatten(1)


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
        self.t = len(self.fps)
        self.n_out = n_out
        self.cy = cy
        self.cx = cx
        r2 = math.sqrt(2.0)
        self.mix = init_(nn.Conv2d(self.t * 3 * PROJ, self.t * n_out * hid, 1, groups=self.t), r2)
        self.out = init_(nn.Conv2d(self.t * n_out * hid, self.t * n_out, 1, groups=self.t * n_out), 0.01)

    def forward(self, q):
        b, _, h, w = q.shape
        feats = torch.cat([pool3(q, dy, dx, self.cy, self.cx) for dy, dx in self.fps], 1)
        o = self.out(F.relu(self.mix(feats)))
        return o.view(b, self.t, self.n_out, h, w)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.proj = init_(nn.Conv2d(WIDTH + SKIP, PROJ, 1), math.sqrt(2.0))
        self.nh = 2 * NORD
        self.head = TypeHeads(spec.footprints, n_out=self.nh, hid=HID)
        self.dense = init_(nn.Linear(GH, spec.n_actions), 0.01)
        self.hyp = nn.Parameter(torch.zeros(self.nh))

    def forward(self, obs, mask):
        u, g = self.trunk(obs)
        p = self.proj(u)
        tb = self.trunk.tbias(g)[:, :, None, None]
        d = self.trunk.done(g)
        o = self.head(p)
        w = F.softmax(5.0 * self.hyp, 0)
        logits = None
        for j in range(self.nh):
            flat = order_flat(o[:, :, j] + tb, j % NORD)
            if j < NORD:
                full = torch.cat([flat, d], 1)
            else:
                full = torch.cat([d, flat], 1)
            term = w[j] * full
            logits = term if logits is None else logits + term
        logits = logits + self.dense(g)
        return logits, self.trunk.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
