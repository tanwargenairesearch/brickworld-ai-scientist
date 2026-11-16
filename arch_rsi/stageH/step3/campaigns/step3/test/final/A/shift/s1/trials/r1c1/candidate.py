import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
PROJ = 16
HID = 12
GLOB = 64
SCALE = 5.0
R2 = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def frame_list(Y, X):
    ts = [False, True] if Y == X else [False]
    out = []
    for t in ts:
        for fy in (False, True):
            for fx in (False, True):
                out.append((t, fy, fx))
    return out


def to_frame(z, fr):
    t, fy, fx = fr
    if t:
        z = z.transpose(-1, -2)
    dims = []
    if fy:
        dims.append(-2)
    if fx:
        dims.append(-1)
    if len(dims) > 0:
        z = torch.flip(z, dims)
    return z


def stacked_ortho(n, rows, cols, gain):
    ws = []
    for _ in range(n):
        w = torch.empty(rows, cols)
        nn.init.orthogonal_(w, gain)
        ws.append(w)
    return torch.stack(ws, 0)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = ortho(nn.Conv2d(cin, STEM, 1), R2)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), R2)

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1), x


class FramePoolHead(nn.Module):
    def __init__(self, cin, spec, proj, hid, per_type, scale):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.frames = frame_list(spec.Y, spec.X)
        G = len(self.frames)
        T = len(self.fps)
        self.scale = scale
        self.per_type = per_type
        self.proj = ortho(nn.Conv2d(cin, proj, 1), 1.0)
        self.w1 = nn.Parameter(stacked_ortho(G * T, hid, 3 * proj, R2).view(G, T, hid, 3 * proj))
        self.b1 = nn.Parameter(torch.zeros(G, T, hid, 1))
        self.w2 = nn.Parameter(stacked_ortho(G * T, 1, hid, 0.01).view(G, T, hid))
        self.b2 = nn.Parameter(torch.zeros(G, T, 1))
        if per_type:
            self.mix = nn.Parameter(torch.zeros(T, G))
        else:
            self.mix = nn.Parameter(torch.zeros(G))

    def pool(self, pg):
        res = []
        for dy, dx in self.fps:
            if dy * dx == 1:
                res.append(torch.cat([pg, pg, pg], 1))
            else:
                q = F.pad(pg, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
                res.append(torch.cat([mx, mn, av], 1))
        return torch.stack(res, 1)

    def forward(self, f):
        p = self.proj(f)
        q = torch.stack([self.pool(to_frame(p, fr)) for fr in self.frames], 1)
        q = q.flatten(4)
        h = F.relu(torch.einsum('bgtcn,gthc->bgthn', q, self.w1) + self.b1)
        l = torch.einsum('bgthn,gth->bgtn', h, self.w2) + self.b2
        if self.per_type:
            w = F.softmax(self.scale * self.mix, dim=1)
            out = (l * w.t().unsqueeze(0).unsqueeze(-1)).sum(1)
        else:
            w = F.softmax(self.scale * self.mix, dim=0)
            out = (l * w.view(1, -1, 1, 1)).sum(1)
        return out.reshape(out.shape[0], -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = len(spec.footprints)
        self.trunk = Trunk(spec.in_channels)
        self.head = FramePoolHead(WIDTH + SKIP, spec, PROJ, HID, False, SCALE)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * WIDTH, GLOB), R2), nn.ReLU())
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.type_bias = ortho(nn.Linear(GLOB, self.T), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        f, x = self.trunk(obs)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        place = self.head(f)
        B = place.shape[0]
        place = (place.reshape(B, self.T, -1) + self.type_bias(g).unsqueeze(-1)).reshape(B, -1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
