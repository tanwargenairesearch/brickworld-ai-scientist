import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'ppo_epochs': 2, 'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 16, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.005}

STEM = 64
COL = 48
WIDTH = 36
SKIP = 16
SQ = 32
GH = 64


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def tf(x, k):
    if k & 4:
        x = x.transpose(2, 3)
    if k & 1:
        x = x.flip(2)
    if k & 2:
        x = x.flip(3)
    return x


class Trunk(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        r2 = math.sqrt(2.0)
        self.s1 = init_(nn.Conv2d(c, STEM, 1), r2)
        self.s2 = init_(nn.Conv2d(STEM, COL, 1), r2)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), r2)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), r2)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.skip = init_(nn.Conv2d(c, SKIP, 1), r2)
        self.glob = nn.Sequential(init_(nn.Linear(2 * WIDTH, GH), r2), nn.ReLU())
        self.done = init_(nn.Linear(GH, 1), 0.01)
        self.value = init_(nn.Linear(GH, 1), 1.0)
        self.tbias = init_(nn.Linear(GH, spec.n_types), 0.01)

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        u = torch.cat([f, F.relu(self.skip(obs))], 1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        return u, g


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.trunk = Trunk(spec)
        t = spec.n_types
        r2 = math.sqrt(2.0)
        self.k = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.frames = [0, 4]
        self.sqs = nn.ModuleList([init_(nn.Conv2d(WIDTH + SKIP, SQ, self.k), r2) for _ in self.frames])
        self.outs = nn.ModuleList([init_(nn.Conv2d(SQ, t, 1), 0.01) for _ in self.frames])
        self.tmix = nn.Parameter(torch.zeros(len(self.frames), t, t))
        self.dense = init_(nn.Linear(GH, spec.n_actions), 0.01)
        self.hyp = nn.Parameter(torch.zeros(len(self.frames)))

    def forward(self, obs, mask):
        u, g = self.trunk(obs)
        tb = self.trunk.tbias(g)[:, :, None, None]
        w = F.softmax(5.0 * self.hyp, 0)
        place = None
        for i, k in enumerate(self.frames):
            q = F.pad(tf(u, k), (0, self.k - 1, 0, self.k - 1))
            h = F.relu(self.sqs[i](q))
            m = self.outs[i](h) + tb
            m = m + torch.einsum('ts,bshw->bthw', self.tmix[i], m)
            term = w[i] * m.flatten(1)
            place = term if place is None else place + term
        logits = torch.cat([place, self.trunk.done(g)], 1) + self.dense(g)
        return logits, self.trunk.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
