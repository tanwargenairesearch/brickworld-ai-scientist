import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
RED = 16
HFC = 384


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
        self.fc1 = nn.Linear(RED * cells, HFC)
        self.fc2 = nn.Linear(HFC, spec.n_place)
        nn.init.orthogonal_(self.fc2.weight, 0.01)
        nn.init.zeros_(self.fc2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + HFC, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.fc1(F.relu(self.red(f)).flatten(1)))
        place = self.fc2(h)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), h], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
