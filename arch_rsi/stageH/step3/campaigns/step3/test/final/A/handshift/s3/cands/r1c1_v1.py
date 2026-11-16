import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
GLOB = 64
PROJ = 16
HID = 12
SCALE = 5.0
PER_TYPE = False
R2 = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def apply_frame(t, g):
    if g >= 4:
        t = t.transpose(-1, -2)
    k = g % 4
    if k == 1:
        t = t.flip(-1)
    elif k == 2:
        t = t.flip(-2)
    elif k == 3:
        t = t.flip(-2).flip(-1)
    return t


class Trunk(nn.Module):
    def __init__(self, spec, width, skip):
        super().__init__()
        c = spec.in_channels
        self.stem1 = ortho(nn.Conv2d(c, STEM, 1), R2)
        self.stem2 = ortho(nn.Conv2d(STEM, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, width, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(4, width)
        self.c2 = ortho(nn.Conv2d(width, width, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(4, width)
        self.skip = ortho(nn.Conv2d(c, skip, 1), 1.0)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * width, GLOB), R2), nn.ReLU())
        self.out_ch = width + skip

    def forward(self, obs):
        s = F.relu(self.stem1(obs))
        s = F.relu(self.stem2(s))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, self.skip(obs)], dim=1)
        return h, g


class FramePoolHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.nf = 8 if self.Y == self.X else 4
        self.proj = ortho(nn.Conv2d(cin, PROJ, 1), R2)
        G = self.nf * self.T
        w1 = torch.empty(G, HID, 3 * PROJ)
        w2 = torch.empty(G, 1, HID)
        for i in range(G):
            nn.init.orthogonal_(w1[i], R2)
            nn.init.orthogonal_(w2[i], 0.01)
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(torch.zeros(G, HID, 1))
        self.w2 = nn.Parameter(w2)
        self.b2 = nn.Parameter(torch.zeros(G, 1, 1))
        if PER_TYPE:
            self.alpha = nn.Parameter(torch.zeros(self.T, self.nf))
        else:
            self.alpha = nn.Parameter(torch.zeros(self.nf))

    def forward(self, h):
        B = h.shape[0]
        nf, T, Y, X = self.nf, self.T, self.Y, self.X
        YX = Y * X
        p = self.proj(h)
        frames = torch.stack([apply_frame(p, g) for g in range(nf)], 0)
        q = frames.reshape(nf * B, PROJ, frames.shape[-2], frames.shape[-1])
        feats = []
        for dy, dx in self.fps:
            if dy * dx == 1:
                feats.append(torch.cat([q, q, q], dim=1))
            else:
                qq = F.pad(q, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(qq, (dy, dx), stride=1)
                mn = -F.max_pool2d(-qq, (dy, dx), stride=1)
                av = F.avg_pool2d(qq, (dy, dx), stride=1)
                feats.append(torch.cat([mx, mn, av], dim=1))
        z = torch.stack(feats, 1)
        z = z.reshape(nf, B, T, 3 * PROJ, YX).permute(0, 2, 3, 1, 4).reshape(nf * T, 3 * PROJ, B * YX)
        hd = F.relu(torch.bmm(self.w1, z) + self.b1)
        out = torch.bmm(self.w2, hd) + self.b2
        out = out.reshape(nf, T, B, YX).permute(2, 0, 1, 3)
        if PER_TYPE:
            w = torch.softmax(SCALE * self.alpha, dim=1).t()
            res = (out * w.reshape(1, nf, T, 1)).sum(1)
        else:
            w = torch.softmax(SCALE * self.alpha, dim=0)
            res = (out * w.reshape(1, nf, 1, 1)).sum(1)
        return res


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec, WIDTH, SKIP)
        self.head = FramePoolHead(self.trunk.out_ch, spec)
        self.T = len(spec.footprints)
        self.type_bias = ortho(nn.Linear(GLOB, self.T), 0.01)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        place = self.head(h)
        place = place + self.type_bias(g).unsqueeze(2)
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
