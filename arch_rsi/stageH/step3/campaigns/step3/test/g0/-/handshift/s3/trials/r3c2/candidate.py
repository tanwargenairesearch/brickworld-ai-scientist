import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001}

H1 = 320
H2 = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nin = int(spec.in_channels) * int(spec.Y) * int(spec.X) + int(spec.n_actions)
        self.l1 = nn.Linear(nin, H1)
        self.l2 = nn.Linear(H1, H2)
        self.pi = nn.Linear(H2, int(spec.n_actions))
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(H2, 1)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.to(obs.dtype)], dim=1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
