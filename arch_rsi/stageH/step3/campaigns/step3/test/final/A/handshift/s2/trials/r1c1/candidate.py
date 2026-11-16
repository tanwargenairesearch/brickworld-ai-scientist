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
HID = 12
GH = 64
SCALE = 5.0
PER_TYPE = False


def frame_list(Y, X):
    trs = [False, True] if Y == X else [False]
    out = []
    for tr in trs:
        for fy in (False, True):
            for fx in (False, True):
                out.append((tr, fy, fx))
    return out


def to_frame(t, fr):
    tr, fy, fx = fr
    if tr:
        t = t.transpose(-1, -2)
    dims = []
    if fy:
        dims.append(-2)
    if fx:
        dims.append(-1)
    if len(dims) > 0:
        t = torch.flip(t, dims)
    return t


def fp_pool(q, dy, dx):
    if dy * dx == 1:
        return q, q, q
    q = F.pad(q, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, F.relu(self.skip(obs))], dim=1)
        return x, f


class FrameFootHead(nn.Module):
    def __init__(self, c, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.frames = frame_list(int(spec.Y), int(spec.X))
        self.nt = len(self.fps)
        nf = len(self.frames)
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        a = self.nt if PER_TYPE else 1
        self.alpha = nn.Parameter(torch.zeros(a, nf))

    def forward(self, f):
        B = f.shape[0]
        nf = len(self.frames)
        p = self.proj(f)
        P = torch.cat([to_frame(p, fr) for fr in self.frames], dim=0)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = fp_pool(P, dy, dx)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        L = torch.cat(res, dim=1)
        L = L.reshape(nf, B, self.nt, -1)
        w = torch.softmax(SCALE * self.alpha, dim=-1).t()
        return (L * w[:, None, :, None]).sum(0)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        self.head = FrameFootHead(WIDTH + SKIP, spec)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.tbias = nn.Linear(GH, spec.n_types)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.head.outs:
            nn.init.orthogonal_(m.weight, 0.01)
        nn.init.orthogonal_(self.tbias.weight, 0.01)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        x, f = self.trunk(obs)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        pl = self.head(f) + self.tbias(g).unsqueeze(-1)
        logits = torch.cat([pl.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
