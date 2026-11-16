import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.97}

HID = 320


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.inp = nn.Linear(d_in, HID)
        self.h1 = nn.Linear(HID, HID)
        self.h2 = nn.Linear(HID, HID)
        self.pi = nn.Linear(HID, int(spec.n_actions))
        self.v = nn.Linear(HID, 1)
        nn.init.zeros_(self.pi.bias)
        with torch.no_grad():
            self.pi.weight.mul_(0.1)

    def forward(self, obs, mask):
        h = F.relu(self.inp(obs.flatten(1)))
        h = F.relu(h + self.h1(h))
        h = F.relu(h + self.h2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
