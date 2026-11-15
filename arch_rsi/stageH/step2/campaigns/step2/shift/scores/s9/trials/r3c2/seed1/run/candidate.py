import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_obs = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.obs_fc = nn.Linear(n_obs, HID)
        self.mask_fc = nn.Linear(int(spec.n_actions), HID)
        self.fc2 = nn.Linear(HID, HID)
        self.pi = nn.Linear(HID, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        h = F.relu(self.obs_fc(obs.flatten(1)) + self.mask_fc(mask.float()))
        h = F.relu(self.fc2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
