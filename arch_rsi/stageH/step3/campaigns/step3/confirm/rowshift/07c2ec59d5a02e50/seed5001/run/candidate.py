import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"gamma": 0.99, "ent_coef": 0.02, "ppo_epochs": 2}

STEM = 64
WIDTH = 48
WH = 24
GH = 64
DONE_SCALE = 0.1
DONE_INIT = -2.0


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_place = int(spec.n_place)
        n_types = len(spec.footprints)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ph = nn.Conv2d(WIDTH, n_types * WH, 1)
        self.po = nn.Conv2d(n_types * WH, n_types, 5, padding=2, groups=n_types)
        nn.init.normal_(self.po.weight, std=0.01)
        nn.init.zeros_(self.po.bias)
        self.abias = nn.Parameter(torch.zeros(n_place))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.constant_(self.done_head.bias, DONE_INIT / DONE_SCALE)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.ph(f))
        place = self.po(h).flatten(1) + self.abias
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g) * DONE_SCALE
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
