import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1e-3, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
RED = 16
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cells = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.red = nn.Conv2d(WIDTH, RED, 1)
        self.fc = nn.Sequential(nn.Linear(RED * cells, HID), nn.ReLU())
        self.place = nn.Linear(HID, int(spec.n_place))
        with torch.no_grad():
            self.place.weight.mul_(0.01)
            self.place.bias.zero_()
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64 + HID, 1)
        self.value_head = nn.Linear(64 + HID, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        r = F.relu(self.red(f)).flatten(1)
        h = self.fc(r)
        place = self.place(h)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        gh = torch.cat([g, h], dim=1)
        logits = torch.cat([place, self.done_head(gh)], dim=1)
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
