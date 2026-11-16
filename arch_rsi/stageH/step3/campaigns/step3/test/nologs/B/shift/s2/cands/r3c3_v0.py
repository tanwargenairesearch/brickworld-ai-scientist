import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "lr": 0.0007}

HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = spec.in_channels * spec.Y * spec.X + spec.n_actions
        self.fc1 = nn.Linear(n_in, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.ln1 = nn.LayerNorm(HID)
        self.ln2 = nn.LayerNorm(HID)
        self.pi = nn.Linear(HID, spec.n_actions)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.float()], dim=1)
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h))) + h
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
