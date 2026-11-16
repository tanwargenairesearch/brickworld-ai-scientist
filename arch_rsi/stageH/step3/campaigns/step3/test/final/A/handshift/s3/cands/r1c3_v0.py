import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
GLOB = 64
HID = 32
SCALE = 5.0
R2 = math.sqrt(2.0)


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


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


class SquareHeads(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = len(fps)
        self.K = max(max(a, b) for a, b in fps)
        self.sq = nn.ModuleList([ortho(nn.Conv2d(cin, HID, self.K), R2) for _ in range(2)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, self.T, 1), 0.01) for _ in range(2)])
        self.tmix = nn.Parameter(torch.zeros(self.T, self.T))
        self.alpha = nn.Parameter(torch.zeros(2))
        self.dense = ortho(nn.Linear(GLOB, self.T * int(spec.Y) * int(spec.X)), 0.01)

    def forward(self, h, g):
        B = h.shape[0]
        K = self.K
        L = []
        for i in range(2):
            hf = h if i == 0 else h.transpose(-1, -2)
            hf = F.pad(hf, (0, K - 1, 0, K - 1))
            m = self.outs[i](F.relu(self.sq[i](hf)))
            m = m + torch.einsum('st,btyx->bsyx', self.tmix, m)
            L.append(m.reshape(B, -1))
        L = torch.stack(L, 1)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        place = (L * w.reshape(1, 2, 1)).sum(1)
        return place + self.dense(g)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec, WIDTH, SKIP)
        self.head = SquareHeads(self.trunk.out_ch, spec)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        h, g = self.trunk(obs)
        place = self.head(h, g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
