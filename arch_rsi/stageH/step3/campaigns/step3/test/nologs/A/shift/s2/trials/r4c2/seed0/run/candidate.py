import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.001, "gamma": 0.97, "gae_lambda": 0.95, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16}

C0 = 32
C1 = 16
H1 = 384
H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.c0 = nn.Conv2d(spec.in_channels, C0, 1)
        self.c1 = nn.Conv2d(C0, C1, 3, padding=1)
        self.fc1 = nn.Linear(C1 * n, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.pi = nn.Linear(H2, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H2, 1)

    def forward(self, obs, mask):
        x = F.relu(self.c0(obs))
        x = F.relu(self.c1(x))
        h = F.relu(self.fc1(x.flatten(1)))
        h = F.relu(self.fc2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
