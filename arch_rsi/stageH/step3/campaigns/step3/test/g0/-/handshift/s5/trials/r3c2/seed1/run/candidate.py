import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.01}

H1 = 256
H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nin = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.l1 = nn.Linear(nin, H1)
        self.l2 = nn.Linear(H1, H2)
        self.l3 = nn.Linear(H2, H2)
        self.pi = nn.Linear(H2, spec.n_actions)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H2, 1)

    def forward(self, obs, mask):
        x = obs.flatten(1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(h + self.l3(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
