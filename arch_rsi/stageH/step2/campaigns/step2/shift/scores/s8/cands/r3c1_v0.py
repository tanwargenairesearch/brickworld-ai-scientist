import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
COL = 48
WIDTH = 40
RED = 24
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH, RED, 1)
        self.fc = nn.Sequential(nn.Linear(RED * n + 2 * WIDTH, HID), nn.ReLU())
        self.pi = nn.Linear(HID, spec.n_actions)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        self.vfc = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.value_head = nn.Linear(64 + HID, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.red(f)).flatten(1)
        gl = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        h = self.fc(torch.cat([r, gl], dim=1))
        logits = self.pi(h)
        v = self.value_head(torch.cat([h, self.vfc(gl)], dim=1)).squeeze(1)
        return logits, v


def build_policy(spec):
    return Net(spec)
