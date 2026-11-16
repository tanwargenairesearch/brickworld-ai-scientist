import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.005}

H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_place = int(spec.n_place)
        nin = int(spec.in_channels) * int(spec.Y) * int(spec.X) + self.n_place
        self.inp = nn.Linear(nin, H)
        self.n1 = nn.LayerNorm(H)
        self.fc2 = nn.Linear(H, H)
        self.n2 = nn.LayerNorm(H)
        self.pi = nn.Linear(H, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v1 = nn.Linear(H, 64)
        self.v2 = nn.Linear(64, 1)

    def forward(self, obs, mask):
        m = mask[:, :self.n_place].to(obs.dtype)
        z = torch.cat([obs.flatten(1), m], dim=1)
        h = F.relu(self.n1(self.inp(z)))
        h = h + F.relu(self.n2(self.fc2(h)))
        logits = self.pi(h)
        value = self.v2(F.relu(self.v1(h))).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
