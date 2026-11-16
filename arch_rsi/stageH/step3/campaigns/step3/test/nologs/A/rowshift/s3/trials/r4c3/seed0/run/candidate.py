import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.0005, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.l1 = nn.Linear(n_in, HID)
        self.l2 = nn.Linear(HID, HID)
        self.pol = nn.Linear(HID, int(spec.n_actions))
        self.value_head = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.l1.weight, gain=math_sqrt2())
        nn.init.zeros_(self.l1.bias)
        nn.init.orthogonal_(self.l2.weight, gain=math_sqrt2())
        nn.init.zeros_(self.l2.bias)
        nn.init.orthogonal_(self.pol.weight, gain=0.01)
        nn.init.zeros_(self.pol.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        h = F.relu(self.l1(obs.flatten(1)))
        h = F.relu(self.l2(h))
        return self.pol(h), self.value_head(h).squeeze(1)


def math_sqrt2():
    return 1.4142135623730951


def build_policy(spec):
    return Net(spec)
