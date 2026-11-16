import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
W = 40
RED = 24
MEMB = 64
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        cells = int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(C, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.red = nn.Conv2d(2 * W, RED, 1)
        self.memb = nn.Linear(na, MEMB)
        self.fc = nn.Sequential(
            nn.Linear(RED * cells + MEMB, HID), nn.ReLU(),
            nn.Linear(HID, HID), nn.ReLU())
        self.pi = nn.Linear(HID, na)
        self.v = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        s = self.stem(obs)
        t = F.relu(self.c1(s))
        t = F.relu(t + self.c2(t))
        r = F.relu(self.red(torch.cat([s, t], dim=1))).flatten(1)
        m = F.relu(self.memb(mask.to(obs.dtype)))
        h = self.fc(torch.cat([r, m], dim=1))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
