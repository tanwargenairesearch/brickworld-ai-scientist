import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.005}

H = 320


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = int(spec.in_channels) * int(spec.Y) * int(spec.X) + int(spec.n_actions)
        self.l1 = nn.Linear(n_in, H)
        self.l2 = nn.Linear(H, H)
        self.pi = nn.Linear(H, int(spec.n_actions))
        self.v = nn.Linear(H, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        z = torch.cat([obs.flatten(1), mask.float()], dim=1)
        h = F.relu(self.l1(z))
        h = F.relu(h + self.l2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
