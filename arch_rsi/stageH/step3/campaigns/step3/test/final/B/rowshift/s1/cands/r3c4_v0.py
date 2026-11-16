import torch
import torch.nn as nn
import torch.nn.functional as F

HID_MAX = 320
PARAM_CAP = 850000
RECIPE = {"lr": 1e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_actions = int(spec.n_actions)
        din = int(spec.in_channels) * int(spec.Y) * int(spec.X) + self.n_actions
        hid = max(64, min(HID_MAX, PARAM_CAP // (din + HID_MAX + self.n_actions + 2)))
        self.fc1 = nn.Linear(din, hid)
        self.ln1 = nn.LayerNorm(hid)
        self.fc2 = nn.Linear(hid, hid)
        self.ln2 = nn.LayerNorm(hid)
        self.pi = nn.Linear(hid, self.n_actions)
        self.v = nn.Linear(hid, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.float()], 1)
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(h + self.fc2(h)))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
