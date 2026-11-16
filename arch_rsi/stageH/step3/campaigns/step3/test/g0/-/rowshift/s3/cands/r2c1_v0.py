import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

RED1 = 32
RED2 = 12
HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        yx = int(spec.Y) * int(spec.X)
        self.red1 = nn.Conv2d(spec.in_channels, RED1, 1)
        self.red2 = nn.Conv2d(RED1, RED2, 1)
        self.fc1 = nn.Linear(RED2 * yx, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.pi = nn.Linear(HID, spec.n_actions)
        self.v = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        h = F.relu(self.red1(obs))
        h = F.relu(self.red2(h))
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        logits = self.pi(h)
        value = self.v(h).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
