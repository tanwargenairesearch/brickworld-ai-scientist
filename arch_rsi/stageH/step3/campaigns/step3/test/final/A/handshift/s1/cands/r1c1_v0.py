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
SQ2 = math.sqrt(2.0)


def frame(x, g):
    if g >= 4:
        x = x.transpose(-1, -2)
    if g % 2 == 1:
        x = x.flip(-1)
    if (g // 2) % 2 == 1:
        x = x.flip(-2)
    return x


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = ortho(nn.Conv2d(cin, STEM, 1), SQ2)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), SQ2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), SQ2)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), SQ2)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), SQ2)
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        return torch.cat([x, F.relu(self.skip(obs))], dim=1)


class FramePoolHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.nf = 8 if spec.X == spec.Y else 4
        self.proj = ortho(nn.Conv2d(cin, PROJ, 1), SQ2)
        w1 = torch.empty(T, self.nf, 3 * PROJ, HID)
        w2 = torch.empty(T, self.nf, HID)
        for t in range(T):
            nn.init.orthogonal_(w2[t], 0.01)
            for g in range(self.nf):
                nn.init.orthogonal_(w1[t, g], SQ2)
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(torch.zeros(T, self.nf, HID))
        self.w2 = nn.Parameter(w2)
        self.b2 = nn.Parameter(torch.zeros(T, self.nf))
        if PER_TYPE:
            self.alpha = nn.Parameter(torch.zeros(T, self.nf))
        else:
            self.alpha = nn.Parameter(torch.zeros(self.nf))

    def forward(self, f, tb):
        B = f.shape[0]
        p = self.proj(f)
        q = torch.stack([frame(p, g) for g in range(self.nf)], 0)
        nf, _, P, H, W = q.shape
        q = q.reshape(nf * B, P, H, W)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                z = torch.cat([q, q, q], 1)
            else:
                qp = F.pad(q, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(qp, (dy, dx), stride=1)
                mn = -F.max_pool2d(-qp, (dy, dx), stride=1)
                av = F.avg_pool2d(qp, (dy, dx), stride=1)
                z = torch.cat([mx, mn, av], 1)
            z = z.reshape(nf, B, 3 * P, H * W)
            h = F.relu(torch.einsum('fbcn,fch->fbhn', z, self.w1[t]) + self.b1[t][:, None, :, None])
            o = torch.einsum('fbhn,fh->fbn', h, self.w2[t]) + self.b2[t][:, None, None]
            outs.append(o)
        L = torch.stack(outs, 2) + tb[None, :, :, None]
        if PER_TYPE:
            w = torch.softmax(SCALE * self.alpha, dim=1).t()
            L = (L * w[:, None, :, None]).sum(0)
        else:
            w = torch.softmax(SCALE * self.alpha, dim=0)
            L = (L * w[:, None, None, None]).sum(0)
        return L.reshape(B, -1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T = len(spec.footprints)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.head = FramePoolHead(c, spec)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * c, GH), SQ2), nn.ReLU())
        self.done_head = ortho(nn.Linear(GH, 1), 0.01)
        self.value_head = ortho(nn.Linear(GH, 1), 1.0)
        self.tbias = ortho(nn.Linear(GH, T), 0.01)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(f, self.tbias(g))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
