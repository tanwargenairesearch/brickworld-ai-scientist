import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 32
SKIP = 16
SHID = 40
GH = 64
SCALE = 5.0
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


class SquareHead(nn.Module):
    def __init__(self, cin, spec):
        super().__init__()
        self.T = len(spec.footprints)
        self.K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.frames = [0, 4]
        self.convs = nn.ModuleList([ortho(nn.Conv2d(cin, SHID, self.K), SQ2) for _ in self.frames])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(SHID, self.T, 1), 0.01) for _ in self.frames])
        self.tmix = nn.Parameter(torch.zeros(len(self.frames), self.T, self.T))
        self.alpha = nn.Parameter(torch.zeros(len(self.frames)))

    def forward(self, f, tb):
        B = f.shape[0]
        K = self.K
        opts = []
        for i, g in enumerate(self.frames):
            x = frame(f, g)
            x = F.pad(x, (0, K - 1, 0, K - 1))
            m = self.outs[i](F.relu(self.convs[i](x))) + tb[:, :, None, None]
            m = m.reshape(B, self.T, -1)
            m = m + torch.einsum('btn,ts->bsn', m, self.tmix[i])
            opts.append(m.reshape(B, -1))
        L = torch.stack(opts, 0)
        w = torch.softmax(SCALE * self.alpha, dim=0)
        return (L * w[:, None, None]).sum(0)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T = len(spec.footprints)
        self.trunk = Trunk(spec.in_channels)
        c = self.trunk.out_ch
        self.head = SquareHead(c, spec)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * c, GH), SQ2), nn.ReLU())
        self.done_head = ortho(nn.Linear(GH, 1), 0.01)
        self.value_head = ortho(nn.Linear(GH, 1), 1.0)
        self.tbias = ortho(nn.Linear(GH, T), 0.01)
        self.dense = ortho(nn.Linear(GH, spec.n_actions), 0.01)

    def forward(self, obs, mask):
        f = self.trunk(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(f, self.tbias(g))
        logits = torch.cat([place, self.done_head(g)], dim=1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
