import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1e-3, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
HH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.nk = 8 if self.Y == self.X else 4
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, HH, 3, padding=1)
        self.outs = nn.Conv2d(HH, self.nk * self.T, 1)
        self.gate = nn.Parameter(torch.zeros(self.nk))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + HH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def align(self, m, k):
        if k >= 4:
            m = m.transpose(2, 3)
        if k % 2 == 1:
            m = m.flip(2)
        if (k // 2) % 2 == 1:
            m = m.flip(3)
        return m

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.c3(f))
        maps = self.outs(h).reshape(B, self.nk, self.T, self.Y, self.X)
        w = torch.softmax(self.gate, 0) * self.nk
        place = w[0] * maps[:, 0]
        for k in range(1, self.nk):
            place = place + w[k] * self.align(maps[:, k], k)
        place = place.reshape(B, -1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h.mean((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
