import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        C = int(spec.in_channels)
        na = int(spec.n_actions)
        din = C * Y * X + na
        H = max(64, min(320, 600000 // din))
        self.l1 = nn.Linear(din, H)
        self.n1 = nn.LayerNorm(H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, H)
        self.pi = nn.Linear(H, na)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H, 1)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.to(obs.dtype)], dim=1)
        h = F.relu(self.n1(self.l1(x)))
        h = F.relu(h + self.l2(h))
        h2 = F.relu(self.l3(h))
        return self.pi(h2), self.v(h2).squeeze(1)


def build_policy(spec):
    return Net(spec)
