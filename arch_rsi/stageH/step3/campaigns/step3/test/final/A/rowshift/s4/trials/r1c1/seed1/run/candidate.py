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


def frame(t, k):
    if k >= 4:
        t = t.transpose(-2, -1)
    if k % 2 == 1:
        t = t.flip(-1)
    if (k // 2) % 2 == 1:
        t = t.flip(-2)
    return t


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        g = math.sqrt(2.0)
        self.s1 = ortho(nn.Conv2d(cin, STEM, 1), g)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), g)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), g)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), g)
        self.glob = ortho(nn.Linear(2 * WIDTH, GLOB), g)

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        feat = torch.cat([x, k], 1)
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)))
        return feat, g


class FrameHead(nn.Module):
    def __init__(self, cin, fps, nf):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in fps]
        T = len(self.fps)
        self.nf = nf
        self.proj = ortho(nn.Conv2d(cin, PROJ, 1), 1.0)
        self.w1 = nn.Parameter(torch.randn(nf, T, 3 * PROJ, HID) * math.sqrt(2.0 / (3 * PROJ)))
        self.b1 = nn.Parameter(torch.zeros(nf, T, HID))
        self.w2 = nn.Parameter(torch.randn(nf, T, HID) * (0.01 / math.sqrt(HID)))
        self.b2 = nn.Parameter(torch.zeros(nf, T))

    def forward(self, feat):
        p = self.proj(feat)
        B = p.shape[0]
        pf = torch.cat([frame(p, k) for k in range(self.nf)], 0)
        qs = []
        for dy, dx in self.fps:
            if dy * dx == 1:
                q = torch.cat([pf, pf, pf], 1)
            else:
                pad = F.pad(pf, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(pad, (dy, dx), stride=1)
                mn = -F.max_pool2d(-pad, (dy, dx), stride=1)
                av = F.avg_pool2d(pad, (dy, dx), stride=1)
                q = torch.cat([mx, mn, av], 1)
            qs.append(q.flatten(2))
        q = torch.stack(qs, 1)
        T = q.shape[1]
        q = q.reshape(self.nf, B, T, q.shape[2], q.shape[3])
        h = torch.einsum('fbtcn,ftch->fbthn', q, self.w1) + self.b1[:, None, :, :, None]
        h = F.relu(h)
        out = torch.einsum('fbthn,fth->fbtn', h, self.w2) + self.b2[:, None, :, None]
        return out


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nf = 8 if spec.X == spec.Y else 4
        T = spec.n_types
        self.trunk = Trunk(spec.in_channels)
        self.head = FrameHead(WIDTH + SKIP, spec.footprints, nf)
        self.alpha = nn.Parameter(torch.zeros(nf))
        self.done = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value = ortho(nn.Linear(GLOB, 1), 1.0)
        self.tbias = ortho(nn.Linear(GLOB, T), 0.01)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        out = self.head(feat)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = torch.einsum('f,fbtn->btn', w, out) + self.tbias(g)[:, :, None]
        logits = torch.cat([place.flatten(1), self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
