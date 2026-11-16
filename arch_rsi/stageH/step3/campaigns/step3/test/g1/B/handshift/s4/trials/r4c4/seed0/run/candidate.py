import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.003, "gamma": 0.9, "gae_lambda": 0.85}

HID = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.C, self.Y, self.X = int(spec.in_channels), int(spec.Y), int(spec.X)
        self.NA = int(spec.n_actions)
        din = self.C * self.Y * self.X + self.NA
        h = max(64, min(HID, 600000 // din))
        self.l1 = nn.Linear(din, h)
        self.n1 = nn.LayerNorm(h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, h)
        self.pi = nn.Linear(h, self.NA)
        self.v = nn.Linear(h, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        z = torch.cat([obs.reshape(B, -1), mask.to(obs.dtype)], dim=1)
        z = F.relu(self.n1(self.l1(z)))
        z = F.relu(z + self.l2(z))
        z = F.relu(self.l3(z))
        return self.pi(z), self.v(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
