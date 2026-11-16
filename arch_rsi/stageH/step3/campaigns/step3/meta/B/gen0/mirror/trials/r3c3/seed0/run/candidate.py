import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.97}

STEM = 64
WIDTH = 40
K = 16
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.s1 = nn.Conv2d(spec.in_channels, STEM, 1)
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH + STEM, K, 1)
        self.fc = nn.Linear(K * Y * X, HID)
        self.mfc = nn.Linear(int(spec.n_actions), HID)
        self.h1 = nn.Linear(HID, HID)
        self.pi = nn.Linear(HID, int(spec.n_actions))
        self.v = nn.Linear(HID, 1)
        nn.init.zeros_(self.pi.bias)
        with torch.no_grad():
            self.pi.weight.mul_(0.1)

    def forward(self, obs, mask):
        s = F.relu(self.s1(obs))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.red(torch.cat([s, f], dim=1)))
        m = self.mfc(mask.to(obs.dtype))
        h = F.relu(self.fc(r.flatten(1)) + m)
        h = F.relu(h + self.h1(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
