import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        in_dim = spec.in_channels * spec.Y * spec.X + spec.n_actions
        hid = max(64, min(256, 500000 // in_dim))
        self.fc1 = nn.Linear(in_dim, hid)
        self.ln1 = nn.LayerNorm(hid)
        self.fc2 = nn.Linear(hid, hid)
        self.ln2 = nn.LayerNorm(hid)
        self.fc3 = nn.Linear(hid, hid)
        self.ln3 = nn.LayerNorm(hid)
        self.pi = nn.Linear(hid, spec.n_actions)
        self.v = nn.Linear(hid, 1)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        z = torch.cat([obs.flatten(1), mask.float()], dim=1)
        h = F.relu(self.ln1(self.fc1(z)))
        h = h + F.relu(self.ln2(self.fc2(h)))
        h = h + F.relu(self.ln3(self.fc3(h)))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
