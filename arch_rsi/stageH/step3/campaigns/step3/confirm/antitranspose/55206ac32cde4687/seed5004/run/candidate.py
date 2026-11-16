import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
PROJ = 16
HID = 12
GH = 64
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def frames(f, nf):
    outs = [f, f.flip(3), f.flip(2), f.flip(2, 3)]
    if nf == 8:
        ft = f.transpose(2, 3)
        outs = outs + [ft, ft.flip(3), ft.flip(2), ft.flip(2, 3)]
    return torch.cat(outs, 0)


def foot_pool(p, fps):
    res = []
    for dy, dx in fps:
        if dy * dx == 1:
            mx = p
            mn = p
            av = p
        else:
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
        res.append(torch.cat([mx, mn, av], 1))
    return torch.stack(res, 1)


class Trunk(nn.Module):
    def __init__(self, cin, width):
        super().__init__()
        self.s1 = ortho(nn.Conv2d(cin, STEM, 1), math.sqrt(2))
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), math.sqrt(2))
        self.c1 = ortho(nn.Conv2d(COL, width, 3, padding=1), math.sqrt(2))
        self.n1 = nn.GroupNorm(4, width)
        self.c2 = ortho(nn.Conv2d(width, width, 3, padding=1), math.sqrt(2))
        self.n2 = nn.GroupNorm(4, width)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), 1.0)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * width, GH), math.sqrt(2)), nn.ReLU())

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        feat = torch.cat([f, F.relu(self.skip(obs))], 1)
        return feat, g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(self.fps)
        self.nf = 8 if spec.X == spec.Y else 4
        self.trunk = Trunk(spec.in_channels, WIDTH)
        self.proj = ortho(nn.Conv2d(WIDTH + SKIP, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([ortho(nn.Linear(3 * PROJ, HID), math.sqrt(2)) for _ in self.fps])
        self.outs = nn.ModuleList([ortho(nn.Linear(HID, 1), 0.01) for _ in self.fps])
        self.done = ortho(nn.Linear(GH, 1), 0.01)
        self.value = ortho(nn.Linear(GH, 1), 1.0)
        self.tbias = ortho(nn.Linear(GH, self.T), 0.01)
        self.alpha = nn.Parameter(torch.zeros(self.nf))

    def forward(self, obs, mask):
        B = obs.shape[0]
        feat, g = self.trunk(obs)
        p = self.proj(feat)
        s = foot_pool(frames(p, self.nf), self.fps)
        w1 = torch.stack([m.weight for m in self.mix])
        b1 = torch.stack([m.bias for m in self.mix])
        h = F.relu(torch.einsum('ntcyx,thc->nthyx', s, w1) + b1[None, :, :, None, None])
        w2 = torch.stack([m.weight[0] for m in self.outs])
        b2 = torch.stack([m.bias[0] for m in self.outs])
        L = torch.einsum('nthyx,th->ntyx', h, w2) + b2[None, :, None, None]
        L = L.reshape(self.nf, B, self.T, -1) + self.tbias(g)[None, :, :, None]
        w = torch.softmax(SCALE * self.alpha, dim=0)
        place = torch.einsum('gbtk,g->btk', L, w)
        logits = torch.cat([place.flatten(1), self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
