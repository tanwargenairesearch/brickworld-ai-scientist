import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
PROJ = 32
HID = 32
GHID = 64
MIX_SCALE = 2.0


def make_frames(Y, X):
    frames = []
    trs = [False, True] if Y == X else [False]
    for tr in trs:
        for fy in (False, True):
            for fx in (False, True):
                frames.append((tr, fy, fx))
    return frames


def apply_frame(x, fr):
    tr, fy, fx = fr
    if tr:
        x = x.transpose(-1, -2)
    if fy:
        x = x.flip(-2)
    if fx:
        x = x.flip(-1)
    return x


def init_layers(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = int(spec.in_channels)
        self.stem = nn.Sequential(nn.Conv2d(c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GHID), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, F.relu(self.skip(obs))], dim=1)
        return h, g


class FramePoolHead(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.frames = make_frames(self.Y, self.X)
        shapes = set(self.fps)
        if self.Y == self.X:
            shapes = shapes | set((b, a) for a, b in self.fps)
        self.shapes = sorted(shapes)
        self.sel = []
        for fr in self.frames:
            row = []
            for (dy, dx) in self.fps:
                s = (dx, dy) if fr[0] else (dy, dx)
                row.append(self.shapes.index(s))
            self.sel.append(row)
        self.proj = nn.Conv2d(cin, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.shapes])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.shapes])

    def forward(self, h):
        p = self.proj(h)
        maps = []
        for (dy, dx), mix, out in zip(self.shapes, self.mix, self.outs):
            q = F.pad(p, (dx - 1, dx - 1, dy - 1, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
            maps.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        per_frame = []
        for fi, fr in enumerate(self.frames):
            per_type = []
            for ti, (dy, dx) in enumerate(self.fps):
                m = apply_frame(maps[self.sel[fi][ti]], fr)
                per_type.append(m[:, 0, dy - 1:dy - 1 + self.Y, dx - 1:dx - 1 + self.X].flatten(1))
            per_frame.append(torch.stack(per_type, 1))
        return torch.stack(per_frame, 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.YX = int(spec.Y) * int(spec.X)
        self.trunk = Trunk(spec)
        self.head = FramePoolHead(spec, WIDTH + SKIP)
        self.alpha = nn.Parameter(torch.zeros(self.T, len(self.head.frames)))
        self.done_head = nn.Linear(GHID, 1)
        self.value_head = nn.Linear(GHID, 1)
        self.type_bias = nn.Linear(GHID, self.T)
        init_layers(self)
        for m in self.head.outs:
            init_out(m, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        L = self.head(h)
        w = torch.softmax(MIX_SCALE * self.alpha, 1)
        place = torch.einsum('bgtn,tg->btn', L, w)
        place = place + self.type_bias(g).unsqueeze(2)
        place = place.flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
