import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
PROJ = 32
HID = 32
GH = 64
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


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
        k = 1
        for a, b in spec.footprints:
            k = max(k, int(a), int(b))
        self.K = k
        self.trunk = Trunk(spec.in_channels, STEM, COL, WIDTH, SKIP)
        self.proj = nn.Conv2d(WIDTH + SKIP, PROJ, 1)
        self.sq = nn.ModuleList([nn.Conv2d(PROJ, HID, k) for _ in range(2)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, self.T, 1) for _ in range(2)])
        self.tmix = nn.Parameter(torch.zeros(2, self.T, self.T))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done = nn.Linear(GH, 1)
        self.value = nn.Linear(GH, 1)
        self.dense = nn.Linear(GH, int(spec.n_place))
        self.wf = nn.Parameter(torch.zeros(2))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        for m in list(self.outs) + [self.done, self.dense]:
            ortho(m, 0.01)
        ortho(self.value, 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        x, f = self.trunk(obs)
        p = F.relu(self.proj(f))
        outs = []
        for i in range(2):
            pi = p if i == 0 else p.transpose(2, 3)
            q = F.pad(pi, (0, self.K - 1, 0, self.K - 1))
            h = F.relu(self.sq[i](q))
            L = self.outs[i](h)
            L = L + torch.einsum("ts,bsyx->btyx", self.tmix[i], L)
            outs.append(L.reshape(B, -1))
        L = torch.stack(outs, 0)
        w = torch.softmax(SCALE * self.wf, 0)
        place = torch.einsum("f,fbn->bn", w, L)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        place = place + self.dense(g)
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
