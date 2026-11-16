import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 32, "n_minibatches": 32}

H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.l1 = nn.Linear(n_in, H)
        self.l2 = nn.Linear(H, H)
        self.pi = nn.Linear(H, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H, 1)

    def forward(self, obs, mask):
        h = F.relu(self.l1(obs.flatten(1)))
        h = F.relu(self.l2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
