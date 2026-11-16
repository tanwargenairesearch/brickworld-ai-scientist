import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.01}

STEM = 64
COL = 48
WIDTH = 40
FLATC = 16
FHID = 512


def small_init(m):
    nn.init.normal_(m.weight, std=0.01)
    nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y, X = int(spec.Y), int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fproj = nn.Conv2d(WIDTH, FLATC, 1)
        self.fmlp = nn.Sequential(
            nn.Linear(FLATC * Y * X, FHID), nn.LayerNorm(FHID), nn.ReLU())
        self.pol = nn.Linear(FHID, int(spec.n_actions))
        self.val = nn.Sequential(nn.Linear(FHID, 64), nn.ReLU(), nn.Linear(64, 1))
        small_init(self.pol)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = self.fmlp(F.relu(self.fproj(f)).flatten(1))
        return self.pol(h), self.val(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
