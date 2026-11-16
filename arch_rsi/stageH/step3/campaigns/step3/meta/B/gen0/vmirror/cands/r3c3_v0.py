import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.005}

CP = 16
MH = 128
H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.cell = nn.Conv2d(int(spec.in_channels), CP, 1)
        self.mask_in = nn.Linear(na, MH)
        in_dim = CP * n + MH
        h1 = max(128, min(512, 500000 // in_dim))
        self.body = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, H2), nn.ReLU())
        self.pi = nn.Linear(H2, na)
        self.v = nn.Linear(H2, 1)

    def forward(self, obs, mask):
        c = F.relu(self.cell(obs)).flatten(1)
        m = F.relu(self.mask_in(mask.to(obs.dtype)))
        h = self.body(torch.cat([c, m], dim=1))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
