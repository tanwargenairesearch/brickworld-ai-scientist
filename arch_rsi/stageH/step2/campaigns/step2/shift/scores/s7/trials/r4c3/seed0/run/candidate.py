import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.001, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

E1 = 48
RED = 16
HIDN = 384


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.cell = nn.Sequential(
            nn.Conv2d(spec.in_channels, E1, 1), nn.ReLU(),
            nn.Conv2d(E1, RED, 1), nn.ReLU())
        self.fc = nn.Linear(RED * n, HIDN)
        self.mfc = nn.Linear(na, HIDN)
        self.fc2 = nn.Linear(HIDN, HIDN)
        self.pi = nn.Linear(HIDN, na)
        self.v = nn.Linear(HIDN, 1)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        r = self.cell(obs).flatten(1)
        h = F.relu(self.fc(r) + self.mfc(mask.to(obs.dtype)))
        h = F.relu(h + self.fc2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
