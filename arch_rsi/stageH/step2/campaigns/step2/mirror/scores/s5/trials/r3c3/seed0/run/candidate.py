import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "gamma": 0.97, "ent_coef": 0.005}

H1 = 256
H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X)
        self.body = nn.Sequential(
            nn.Linear(n_in, H1), nn.ReLU(),
            nn.Linear(H1, H2), nn.ReLU())
        self.pi = nn.Linear(H2, spec.n_actions)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.vh = nn.Sequential(nn.Linear(H2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        h = self.body(obs.flatten(1))
        logits = self.pi(h)
        return logits, self.vh(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
