import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
PROJ = 16
HID = 16
GH = 64
SCALE = 2.0


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def frame_list(Y, X):
    fr = []
    for tr in ([False, True] if Y == X else [False]):
        for fy in (False, True):
            for fx in (False, True):
                fr.append((tr, fy, fx))
    return fr


def apply_frame(t, fr):
    tr, fy, fx = fr
    if tr:
        t = t.transpose(2, 3)
    dims = []
    if fy:
        dims.append(2)
    if fx:
        dims.append(3)
    if len(dims) > 0:
        t = torch.flip(t, dims)
    return t


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(c, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, COL, 1)), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = init_layer(nn.Conv2d(c, SKIP, 1))

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        return f, torch.cat([f, self.skip(obs)], dim=1)


class FramePoolHead(nn.Module):
    def __init__(self, spec, cin, per_type=True, scale=SCALE):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = spec.n_types
        self.frames = frame_list(spec.Y, spec.X)
        self.nf = len(self.frames)
        self.per_type = per_type
        self.scale = scale
        self.proj = init_layer(nn.Conv2d(cin, PROJ, 1))
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(3 * PROJ, HID, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        if per_type:
            self.w = nn.Parameter(torch.zeros(self.T, self.nf))
        else:
            self.w = nn.Parameter(torch.zeros(self.nf))

    def forward(self, h, tbias):
        B = h.shape[0]
        p = self.proj(h)
        pf = torch.cat([apply_frame(p, fr) for fr in self.frames], dim=0)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                z = torch.cat([pf, pf, pf], dim=1)
            else:
                q = F.pad(pf, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
                z = torch.cat([mx, mn, av], dim=1)
            res.append(out(F.relu(mix(z))))
        m = torch.cat(res, dim=1)
        m = m + tbias.repeat(self.nf, 1)[:, :, None, None]
        L = m.reshape(self.nf, B, self.T, -1)
        if self.per_type:
            wts = torch.softmax(self.scale * self.w, dim=1)
            o = torch.einsum('tf,fbtn->btn', wts, L)
        else:
            wts = torch.softmax(self.scale * self.w, dim=0)
            o = torch.einsum('f,fbtn->btn', wts, L)
        return o.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, GH)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GH, 1), 0.01)
        self.value_head = init_layer(nn.Linear(GH, 1), 1.0)
        self.type_bias = init_layer(nn.Linear(GH, spec.n_types), 0.01)
        self.head = FramePoolHead(spec, WIDTH + SKIP, per_type=True, scale=SCALE)

    def forward(self, obs, mask):
        f, h = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(h, self.type_bias(g))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
