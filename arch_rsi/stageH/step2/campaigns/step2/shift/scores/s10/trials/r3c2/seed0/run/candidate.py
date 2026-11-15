import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0005, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95}

H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.n_place = int(spec.n_place)
        n_actions = int(spec.n_actions)
        din = int(spec.in_channels) * Y * X + self.n_place
        h1 = max(64, min(320, 650000 // din))
        self.l1 = nn.Linear(din, h1)
        self.n1 = nn.LayerNorm(h1)
        self.l2 = nn.Linear(h1, H2)
        self.n2 = nn.LayerNorm(H2)
        self.pi = nn.Linear(H2, n_actions)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Sequential(nn.Linear(H2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        m = mask[:, :self.n_place].to(obs.dtype)
        x = torch.cat([obs.flatten(1), m], dim=1)
        h = F.relu(self.n1(self.l1(x)))
        h = F.relu(self.n2(self.l2(h)))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
