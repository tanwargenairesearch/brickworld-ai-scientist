import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "lr": 0.0005, "ent_coef": 0.01}

H = 320


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.body = nn.Sequential(nn.Linear(n_in, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.pi = nn.Linear(H, int(spec.n_actions))
        self.value_head = nn.Linear(H, 1)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        h = self.body(obs.flatten(1))
        return self.pi(h), self.value_head(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
