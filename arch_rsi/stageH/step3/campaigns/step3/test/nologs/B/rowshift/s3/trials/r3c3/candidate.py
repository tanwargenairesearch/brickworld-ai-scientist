import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        na = int(spec.n_actions)
        inp = int(spec.in_channels) * int(spec.Y) * int(spec.X) + na
        h = int(800000 / (inp + na + 300))
        h = max(64, min(256, h))
        self.l1 = nn.Linear(inp, h)
        self.n1 = nn.LayerNorm(h)
        self.l2 = nn.Linear(h, h)
        self.n2 = nn.LayerNorm(h)
        self.pi = nn.Linear(h, na)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v1 = nn.Linear(h, 64)
        self.v2 = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.float()], dim=1)
        z = F.relu(self.n1(self.l1(x)))
        z = F.relu(self.n2(self.l2(z)))
        logits = self.pi(z)
        value = self.v2(F.relu(self.v1(z))).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
