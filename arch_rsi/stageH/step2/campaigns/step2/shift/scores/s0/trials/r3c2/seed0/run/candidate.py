import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16,
          "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0007}

H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        in_dim = int(spec.in_channels) * int(spec.Y) * int(spec.X) + int(spec.n_actions)
        h1 = max(64, min(384, 650000 // in_dim))
        self.l1 = nn.Linear(in_dim, h1)
        self.l2 = nn.Linear(h1, H2)
        self.pi = nn.Linear(H2, spec.n_actions)
        self.v = nn.Linear(H2, 1)
        for layer, gain in ((self.l1, math.sqrt(2.0)), (self.l2, math.sqrt(2.0)),
                            (self.pi, 0.01), (self.v, 1.0)):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.float()], dim=1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
