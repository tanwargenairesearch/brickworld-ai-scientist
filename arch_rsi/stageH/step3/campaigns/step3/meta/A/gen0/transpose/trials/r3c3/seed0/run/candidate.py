import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01}

H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.inp = nn.Linear(n_in, H)
        self.hid = nn.Linear(H, H)
        self.pi = nn.Linear(H, int(spec.n_actions))
        self.v = nn.Linear(H, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        x = F.relu(self.inp(obs.flatten(1)))
        x = F.relu(x + self.hid(x))
        return self.pi(x), self.v(x).squeeze(1)


def build_policy(spec):
    return Net(spec)
