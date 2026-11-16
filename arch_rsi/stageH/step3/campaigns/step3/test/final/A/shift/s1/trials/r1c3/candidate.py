import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32,
          "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
HID = 24
GLOB = 64
SCALE = 5.0
R2 = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Trunk(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.s1 = ortho(nn.Conv2d(cin, STEM, 1), R2)
        self.s2 = ortho(nn.Conv2d(STEM, COL, 1), R2)
        self.c1 = ortho(nn.Conv2d(COL, WIDTH, 3, padding=1), R2)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), R2)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), R2)

    def forward(self, obs):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        k = F.relu(self.skip(obs))
        return torch.cat([x, k], 1), x


class SquareHead(nn.Module):
    def __init__(self, cin, spec, hid, scale):
        super().__init__()
        self.T = len(spec.footprints)
        self.K = max(max(int(a), int(b)) for a, b in spec.footprints)
        self.scale = scale
        self.conv = ortho(nn.Conv2d(cin, 2 * hid, self.K), R2)
        self.out = ortho(nn.Conv2d(2 * hid, 2 * self.T, 1), 0.01)
        self.tmix = nn.Parameter(torch.zeros(2, self.T, self.T))
        self.mix = nn.Parameter(torch.zeros(2))

    def forward(self, f):
        k = self.K - 1
        h = F.relu(self.conv(F.pad(f, (0, k, 0, k))))
        m = self.out(h)
        B, _, H, W = m.shape
        m = m.reshape(B, 2, self.T, H, W)
        m = m + torch.einsum('fts,bfsyx->bftyx', self.tmix, m)
        a = m[:, 0].reshape(B, -1)
        b = m[:, 1].transpose(-1, -2).reshape(B, -1)
        w = F.softmax(self.scale * self.mix, dim=0)
        return w[0] * a + w[1] * b


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = len(spec.footprints)
        self.trunk = Trunk(spec.in_channels)
        self.head = SquareHead(WIDTH + SKIP, spec, HID, SCALE)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * WIDTH, GLOB), R2), nn.ReLU())
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.type_bias = ortho(nn.Linear(GLOB, self.T), 0.01)
        self.dense = ortho(nn.Linear(GLOB, spec.n_actions), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        f, x = self.trunk(obs)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        place = self.head(f)
        B = place.shape[0]
        place = (place.reshape(B, self.T, -1) + self.type_bias(g).unsqueeze(-1)).reshape(B, -1)
        logits = torch.cat([place, self.done_head(g)], 1) + self.dense(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
