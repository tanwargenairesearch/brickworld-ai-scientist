import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.001}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GH = 64
PROJ = 16
HID = 12
SCALE = 5.0


def make_frames(Y, X):
    trs = [False, True] if Y == X else [False]
    out = []
    for tr in trs:
        for fy in (False, True):
            for fx in (False, True):
                out.append((tr, fy, fx))
    return out


def apply_frame(t, fr):
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


def init_all(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        feat = torch.cat([f, self.skip(obs)], 1)
        return feat, g


class FramePoolHead(nn.Module):
    def __init__(self, cin, footprints, frames):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.frames = frames
        self.proj = nn.Conv2d(cin, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for f in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for f in self.fps])
        self.alpha = nn.Parameter(torch.zeros(len(frames)))

    def forward(self, feat):
        B = feat.shape[0]
        p = self.proj(feat)
        q = torch.cat([apply_frame(p, fr) for fr in self.frames], 0)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = q, q, q
            else:
                z = F.pad(q, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(z, (dy, dx), stride=1)
                mn = -F.max_pool2d(-z, (dy, dx), stride=1)
                av = F.avg_pool2d(z, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], 1)))
            res.append(out(h))
        L = torch.cat(res, 1)
        G = len(self.frames)
        T = len(self.fps)
        L = L.reshape(G, B, T, -1)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        return torch.einsum('g,gbtn->btn', w, L)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec.in_channels)
        frames = make_frames(int(spec.Y), int(spec.X))
        self.head = FramePoolHead(WIDTH + SKIP, spec.footprints, frames)
        T = len(spec.footprints)
        self.type_bias = nn.Linear(GH, T)
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)
        init_all(self)
        for m in self.head.outs:
            init_out(m, 0.01)
        init_out(self.type_bias, 0.01)
        init_out(self.done_head, 0.01)
        init_out(self.value_head, 1.0)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        place = self.head(feat) + self.type_bias(g).unsqueeze(2)
        logits = torch.cat([place.flatten(1), self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
