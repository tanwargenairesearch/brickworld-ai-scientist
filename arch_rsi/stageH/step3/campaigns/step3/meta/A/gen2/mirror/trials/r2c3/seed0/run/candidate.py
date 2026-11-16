import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005}

WIDTH = 40
SKIP = 24
GH = 64
P = 24
H = 16


def init_all(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_out(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def frame(t, k):
    if k >= 4:
        t = t.transpose(2, 3)
    if k % 2 == 1:
        t = t.flip(3)
    if (k // 2) % 2 == 1:
        t = t.flip(2)
    return t


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(c, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.out_ch = WIDTH + SKIP

    def forward(self, obs):
        s = self.stem(obs)
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        u = torch.cat([x, self.skip(obs)], 1)
        return u, g


class Mix(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))

    def forward(self, parts):
        a = torch.softmax(5.0 * self.w, 0)
        return (torch.stack(parts, -1) * a).sum(-1)


class Head(nn.Module):
    def __init__(self, spec, cin):
        super().__init__()
        self.T = spec.n_types
        self.K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.nf = 8 if spec.Y == spec.X else 4
        self.proj = nn.Conv2d(cin, P, 1)
        self.convs = nn.ModuleList([nn.Conv2d(P, H, self.K) for _ in range(self.nf)])
        self.pmix = nn.ModuleList([nn.Conv2d(2 * P, H, 1) for _ in range(self.nf)])
        self.outs = nn.ModuleList([nn.Conv2d(H, self.T, 1) for _ in range(self.nf)])
        self.mix = Mix(self.nf)

    def init_outputs(self):
        for m in self.outs:
            init_out(m, 0.01)

    def forward(self, u):
        p = F.relu(self.proj(u))
        K = self.K
        parts = []
        for k in range(self.nf):
            q = F.pad(frame(p, k), (0, K - 1, 0, K - 1))
            mx = F.max_pool2d(q, K, stride=1)
            mn = -F.max_pool2d(-q, K, stride=1)
            h = F.relu(self.convs[k](q) + self.pmix[k](torch.cat([mx, mn], 1)))
            parts.append(self.outs[k](h).flatten(1))
        return self.mix(parts)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        self.head = Head(spec, self.trunk.out_ch)
        self.hw = spec.Y * spec.X
        self.tbias = nn.Linear(GH, spec.n_types)
        self.done = nn.Linear(GH, spec.n_actions - spec.n_place)
        self.value = nn.Linear(GH, 1)
        init_all(self)
        init_out(self.tbias, 0.01)
        init_out(self.done, 0.01)
        init_out(self.value, 1.0)
        self.head.init_outputs()

    def forward(self, obs, mask):
        u, g = self.trunk(obs)
        place = self.head(u) + self.tbias(g).repeat_interleave(self.hw, dim=1)
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
