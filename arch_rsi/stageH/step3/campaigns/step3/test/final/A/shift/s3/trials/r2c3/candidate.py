import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.0015}

HID = 320


def ortho_init(net):
    for mod in net.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(mod.weight, math.sqrt(2.0))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def set_gain(mod, gain):
    nn.init.orthogonal_(mod.weight, gain)
    nn.init.zeros_(mod.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_actions = int(spec.n_actions)
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X) + self.n_actions
        self.l1 = nn.Linear(n_in, HID)
        self.n1 = nn.LayerNorm(HID)
        self.l2 = nn.Linear(HID, HID)
        self.l3 = nn.Linear(HID, HID)
        self.pi = nn.Linear(HID, self.n_actions)
        self.value_head = nn.Linear(HID, 1)
        ortho_init(self)
        set_gain(self.pi, 0.01)
        set_gain(self.value_head, 1.0)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.float()], dim=1)
        h = F.relu(self.n1(self.l1(x)))
        h = F.relu(self.l2(h))
        h = h + F.relu(self.l3(h))
        return self.pi(h), self.value_head(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
