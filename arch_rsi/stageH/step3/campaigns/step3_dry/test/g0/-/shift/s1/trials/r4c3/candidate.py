import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97}

STEM = 64
COL = 48
WIDTH = 40
R = 32
AH = 40


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        N = self.Y * self.X
        self.n_place = int(spec.n_place)
        self.T = self.n_place // N
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.rproj = nn.Conv2d(WIDTH, R, 1)
        self.remap = nn.Linear(N, N)
        self.a1 = nn.Conv2d(R + self.T, AH, 3, padding=1)
        self.a2 = nn.Conv2d(AH, AH, 3, padding=1)
        self.a3 = nn.Conv2d(AH, self.T, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * AH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.rproj(f)).flatten(2)
        r = self.remap(r).reshape(B, R, self.Y, self.X)
        m = mask[:, :self.n_place].float().reshape(B, self.T, self.Y, self.X)
        a = F.relu(self.a1(torch.cat([r, m], dim=1)))
        a = F.relu(a + self.a2(a))
        place = self.a3(a).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), a.mean((2, 3)), a.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
