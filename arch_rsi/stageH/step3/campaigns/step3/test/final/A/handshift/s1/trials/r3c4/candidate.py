import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.001}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
WHID = 28
GH = 64
SCALE = 5.0
NORD = 3
SQ2 = 1.4142135623730951


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


class LayoutHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        self.T = len(spec.footprints)
        K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.nf = 8 if spec.X == spec.Y else 4
        self.nopt = self.nf * NORD * 2
        self.win = ortho(nn.Conv2d(cin, WHID, 2 * K - 1, padding=K - 1), SQ2)
        self.out = ortho(nn.Conv2d(WHID, self.nopt * self.T, 1), 0.01)
        self.alpha = nn.Parameter(torch.zeros(self.nopt))

    def forward(self, f, tb, d):
        B, _, Y, X = f.shape
        T = self.T
        M = self.out(F.relu(self.win(f))).reshape(B, self.nopt, T, Y, X)
        M = M + tb[:, None, :, None, None]
        opts = []
        k = 0
        for g in range(self.nf):
            for o in range(NORD):
                for dp in range(2):
                    m = frame(M[:, k], g)
                    if o == 1:
                        m = m.permute(0, 2, 3, 1)
                    elif o == 2:
                        m = m.permute(0, 2, 1, 3)
                    m = m.reshape(B, -1)
                    if dp == 0:
                        opts.append(torch.cat([m, d], 1))
                    else:
                        opts.append(torch.cat([d, m], 1))
                    k += 1
        L = torch.stack(opts, 0)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        return (L * w[:, None, None]).sum(0)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T = len(spec.footprints)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.head = LayoutHead(c, spec)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * c, GH), SQ2), nn.ReLU())
        self.done_head = ortho(nn.Linear(GH, 1), 0.01)
        self.value_head = ortho(nn.Linear(GH, 1), 1.0)
        self.tbias = ortho(nn.Linear(GH, T), 0.01)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = self.head(f, self.tbias(g), self.done_head(g))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
