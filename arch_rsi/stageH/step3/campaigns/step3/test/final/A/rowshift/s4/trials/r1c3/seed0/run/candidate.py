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
SQ = 52
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.frames = [0, 4] if spec.X == spec.Y else [0, 3]
        T = spec.n_types
        self.K = max(max(int(a), int(b)) for a, b in spec.footprints)
        cin = WIDTH + SKIP
        self.trunk = Trunk(spec.in_channels)
        self.convs = nn.ModuleList([ortho(nn.Conv2d(cin, SQ, self.K), math.sqrt(2.0)) for _ in self.frames])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(SQ, T, 1), 0.01) for _ in self.frames])
        self.tmix = nn.ModuleList([nn.Conv2d(T, T, 1, bias=False) for _ in self.frames])
        for m in self.tmix:
            nn.init.zeros_(m.weight)
        self.dense = ortho(nn.Linear(GLOB, spec.n_place), 0.01)
        self.alpha = nn.Parameter(torch.zeros(len(self.frames)))
        self.done = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value = ortho(nn.Linear(GLOB, 1), 1.0)
        self.tbias = ortho(nn.Linear(GLOB, T), 0.01)

    def forward(self, obs, mask):
        feat, g = self.trunk(obs)
        B = obs.shape[0]
        tb = self.tbias(g)[:, :, None, None]
        K = self.K
        comps = []
        for k, conv, out, tm in zip(self.frames, self.convs, self.outs, self.tmix):
            fk = frame(feat, k)
            q = F.pad(fk, (0, K - 1, 0, K - 1))
            L = out(F.relu(conv(q)))
            L = L + tm(L) + tb
            comps.append(L.reshape(B, -1))
        Ls = torch.stack(comps, 1)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = torch.einsum('k,bka->ba', w, Ls) + self.dense(g)
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
