import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}

H = 320


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        na = int(spec.n_actions)
        inp = int(spec.in_channels) * int(spec.Y) * int(spec.X) + na
        self.fc1 = nn.Linear(inp, H)
        self.fc2 = nn.Linear(H, H)
        self.pi = nn.Linear(H, na)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H, 1)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.to(obs.dtype)], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(h + self.fc2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
