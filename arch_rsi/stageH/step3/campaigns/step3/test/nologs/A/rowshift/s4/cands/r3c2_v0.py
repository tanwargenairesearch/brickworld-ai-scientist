import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.98, "gae_lambda": 0.95, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.0007, "ent_coef": 0.01}

H = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nin = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.inp = nn.Linear(nin, H)
        self.minp = nn.Linear(na, H)
        self.h2 = nn.Linear(H, H)
        self.h3 = nn.Linear(H, H)
        self.pi = nn.Linear(H, na)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H, 1)

    def forward(self, obs, mask):
        x = F.relu(self.inp(obs.flatten(1)) + self.minp(mask.float()))
        x = F.relu(self.h2(x))
        x = F.relu(x + self.h3(x))
        return self.pi(x), self.v(x).squeeze(1)


def build_policy(spec):
    return Net(spec)
