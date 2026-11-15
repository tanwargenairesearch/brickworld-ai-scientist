import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_cells = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.fc = nn.Linear(PROJ * n_cells, HID)
        self.gfc = nn.Linear(2 * WIDTH, HID)
        self.pi = nn.Linear(HID, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v1 = nn.Linear(HID, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = F.relu(self.proj(f)).flatten(1)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        h = F.relu(self.fc(p) + self.gfc(g))
        logits = self.pi(h)
        value = self.value_head(F.relu(self.v1(h))).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
