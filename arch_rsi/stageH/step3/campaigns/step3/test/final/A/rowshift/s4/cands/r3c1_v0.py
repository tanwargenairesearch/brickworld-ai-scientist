import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 32
COL = 48
WIDTH = 32
SKIP = 8
GLOB = 64
WIN = 36
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
    def __init__(self, cin, nf):
        super().__init__()
        g = math.sqrt(2.0)
        self.nf = nf
        cx = cin * nf
        self.s1 = ortho(nn.Conv2d(cx, STEM, 1), g)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), g)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), g)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), g)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = ortho(nn.Conv2d(cx, SKIP, 1), g)
        self.glob = ortho(nn.Linear(2 * WIDTH, GLOB), g)

    def forward(self, obs):
        xin = torch.cat([frame(obs, k) for k in range(self.nf)], 1)
        s = F.relu(self.s2(F.relu(self.s1(xin))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(xin))
        feat = torch.cat([x, k], 1)
        g = F.relu(self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)))
        return feat, g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nf = 8 if spec.X == spec.Y else 4
        self.T = spec.n_types
        K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.trunk = Trunk(spec.in_channels, self.nf)
        self.win = ortho(nn.Conv2d(WIDTH + SKIP, WIN, 2 * K - 1, padding=K - 1), math.sqrt(2.0))
        self.out = ortho(nn.Conv2d(WIN, self.nf * self.T, 1), 0.01)
        self.dense = ortho(nn.Linear(GLOB, spec.n_place), 0.01)
        self.alpha = nn.Parameter(torch.zeros(2 * self.nf))
        self.done = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value = ortho(nn.Linear(GLOB, 1), 1.0)
        self.tbias = ortho(nn.Linear(GLOB, self.T), 0.01)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        B = obs.shape[0]
        m = self.out(F.relu(self.win(feat)))
        Y, X = m.shape[2], m.shape[3]
        m = m.reshape(B, self.nf, self.T, Y, X)
        tb = self.tbias(g)[:, :, None, None]
        comps = []
        for k in range(self.nf):
            mk = frame(m[:, k], k) + tb
            comps.append(mk.reshape(B, -1))
            comps.append(mk.reshape(B, self.T, -1).transpose(1, 2).reshape(B, -1))
        L = torch.stack(comps, 1)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = torch.einsum('k,bka->ba', w, L) + self.dense(g)
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
