import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01}

HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.obs_in = nn.Linear(n_in, HID)
        self.mask_in = nn.Linear(int(spec.n_actions), HID)
        self.l2 = nn.Linear(HID, HID)
        self.pi = nn.Linear(HID, int(spec.n_actions))
        nn.init.zeros_(self.pi.bias)
        self.pi.weight.data.mul_(0.1)
        self.v = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        h = F.relu(self.obs_in(obs.flatten(1)) + self.mask_in(mask.float()))
        h = F.relu(self.l2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
