import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2}

H1 = 256
H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nin = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.l1 = nn.Linear(nin, H1)
        self.l2 = nn.Linear(H1, H2)
        self.pi = nn.Linear(H2, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v1 = nn.Linear(H1, 64)
        self.v2 = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.l1(obs.flatten(1)))
        h2 = F.relu(self.l2(h))
        logits = self.pi(h2)
        v = self.v2(F.relu(self.v1(h))).squeeze(1)
        return logits, v


def build_policy(spec):
    return Net(spec)
