import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.0007, "ent_coef": 0.005, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        din = int(spec.in_channels) * n
        na = int(spec.n_actions)
        h = max(64, min(256, 450000 // (din + na + 256)))
        self.l1 = nn.Linear(din, h)
        self.l2 = nn.Linear(h, h)
        self.pi = nn.Linear(h, na)
        self.v = nn.Linear(h, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        x = F.relu(self.l1(obs.flatten(1)))
        x = F.relu(self.l2(x))
        return self.pi(x), self.v(x).squeeze(1)


def build_policy(spec):
    return Net(spec)
