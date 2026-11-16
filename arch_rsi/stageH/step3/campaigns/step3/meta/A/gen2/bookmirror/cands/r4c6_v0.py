import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.002}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 24
PROJ = 16
HID = 16
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


def wpool(q, dy, dx, top=True, left=True):
    if dy * dx == 1:
        return q, q, q
    pad = ((0, dx - 1) if left else (dx - 1, 0)) + ((0, dy - 1) if top else (dy - 1, 0))
    q = F.pad(q, pad)
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


def vkey(t, a, b, top, left):
    if a == 1:
        top = True
    if b == 1:
        left = True
    return (t, a, b, top, left)


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
        self.frames = list(range(8)) if spec.X == spec.Y else [0, 1, 2, 3]
        keys = []
        sel = []
        for k in self.frames:
            tr = k >= 4
            m = k % 4
            fx = m in (1, 3)
            fy = m in (2, 3)
            rt = []
            for t, (dy, dx) in enumerate(self.fps):
                if tr:
                    key = vkey(t, dx, dy, not fx, not fy)
                else:
                    key = vkey(t, dy, dx, not fy, not fx)
                if key not in keys:
                    keys.append(key)
                rt.append(keys.index(key))
            sel.append(rt)
        self.keys = keys
        self.sel = sel
        self.proj = ortho(nn.Conv2d(self.base.ch, PROJ, 1), math.sqrt(2))
        self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * PROJ, HID, 1), math.sqrt(2)) for _ in keys])
        self.out = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for _ in keys])
        self.af = nn.Parameter(torch.zeros(len(self.frames)))

    def forward(self, obs, mask):
        f, g = self.base.features(obs)
        p = self.proj(f)
        B = p.shape[0]
        H = []
        for key, mix, out in zip(self.keys, self.mix, self.out):
            t, a, b, top, left = key
            mx, mn, av = wpool(p, a, b, top, left)
            H.append(out(F.relu(mix(torch.cat([mx, mn, av], 1)))))
        wf = torch.softmax(SCALE * self.af, 0)
        tb = self.base.tbias(g).view(B, self.T, 1, 1)
        place = None
        for i, k in enumerate(self.frames):
            A = torch.cat([H[j] for j in self.sel[i]], 1) + tb
            term = wf[i] * dihedral(A, k).reshape(B, -1)
            place = term if place is None else place + term
        logits = torch.cat([place, self.base.done(g)], 1)
        return logits, self.base.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
