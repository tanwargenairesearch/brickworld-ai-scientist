import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.02}

HID = 256


class MLP(nn.Module):
    def __init__(self, spec):
        super().__init__()
        nin = int(spec.in_channels) * int(spec.Y) * int(spec.X) + int(spec.n_actions)
        self.l1 = nn.Linear(nin, HID)
        self.l2 = nn.Linear(HID, HID)
        self.pi = nn.Linear(HID, spec.n_actions)
        self.v = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        x = torch.cat([obs.reshape(B, -1), mask.to(obs.dtype)], dim=1)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return MLP(spec)
