import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ppo_epochs": 1, "lr": 0.001, "ent_coef": 0.005}

H1 = 384
H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.l1 = nn.Linear(d, H1)
        self.l2 = nn.Linear(H1, H2)
        self.pi = nn.Linear(H2, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H2, 1)

    def forward(self, obs, mask):
        h = F.relu(self.l1(obs.flatten(1)))
        h = F.relu(self.l2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
