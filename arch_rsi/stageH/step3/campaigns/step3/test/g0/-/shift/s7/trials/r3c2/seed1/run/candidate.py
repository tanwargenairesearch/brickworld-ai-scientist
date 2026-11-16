import torch
import torch.nn as nn
import torch.nn.functional as F

HID = 288

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "lr": 0.0005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X) + int(spec.n_actions)
        self.fc1 = nn.Linear(n_in, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.pi = nn.Linear(HID, int(spec.n_actions))
        self.v = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.float()], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
