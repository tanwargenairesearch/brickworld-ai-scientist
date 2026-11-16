import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.003}

STEM = 64
COL = 32
WIDTH = 32
SKIP = 16
PROJ = 16
HID = 16
GH = 64
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def frame(x, i):
    if i >= 4:
        x = x.transpose(2, 3)
    j = i % 4
    if j & 1:
        x = x.flip(3)
    if j & 2:
        x = x.flip(2)
    return x


class Trunk(nn.Module):
    def __init__(self, cin, stem, col, width, skip):
        super().__init__()
        self.s1 = nn.Conv2d(cin, stem, 1)
        self.s2 = nn.Conv2d(stem, col, 1)
        self.c1 = nn.Conv2d(col, width, 3, padding=1)
        self.n1 = nn.GroupNorm(4, width)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.n2 = nn.GroupNorm(4, width)
        self.skip = nn.Conv2d(cin, skip, 1)

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        return x, torch.cat([x, F.relu(self.skip(obs))], 1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.nf = 8 if self.Y == self.X else 4
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.trunk = Trunk(spec.in_channels, STEM, COL, WIDTH, SKIP)
        self.proj = nn.Conv2d(WIDTH + SKIP, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done = nn.Linear(GH, 1)
        self.value = nn.Linear(GH, 1)
        self.tbias = nn.Linear(GH, self.T)
        self.wf = nn.Parameter(torch.zeros(self.nf))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for m in list(self.outs) + [self.done, self.tbias]:
            ortho(m, 0.01)
        ortho(self.value, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        x, f = self.trunk(obs)
        p = self.proj(f)
        pf = torch.cat([frame(p, i) for i in range(self.nf)], 0)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx = pf
                mn = pf
                av = pf
            else:
                q = F.pad(pf, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], 1)))))
        L = torch.cat(res, 1).reshape(self.nf, B, self.T, -1)
        w = torch.softmax(SCALE * self.wf, 0)
        place = torch.einsum("f,fbtn->btn", w, L)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        place = place + self.tbias(g).unsqueeze(2)
        logits = torch.cat([place.reshape(B, -1), self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
