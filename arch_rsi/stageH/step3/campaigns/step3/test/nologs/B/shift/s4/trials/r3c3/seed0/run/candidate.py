import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.01}

HID = 320


def small_init(m):
    nn.init.normal_(m.weight, std=0.01)
    nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.l1 = nn.Linear(n_in, HID)
        self.n1 = nn.LayerNorm(HID)
        self.l2 = nn.Linear(HID, HID)
        self.n2 = nn.LayerNorm(HID)
        self.pol = nn.Linear(HID, int(spec.n_actions))
        self.val = nn.Linear(HID, 1)
        small_init(self.pol)

    def forward(self, obs, mask):
        h = F.relu(self.n1(self.l1(obs.flatten(1))))
        h = h + F.relu(self.n2(self.l2(h)))
        return self.pol(h), self.val(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
