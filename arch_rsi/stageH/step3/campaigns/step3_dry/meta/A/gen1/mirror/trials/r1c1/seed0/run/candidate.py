import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_in = spec.in_channels * spec.Y * spec.X + spec.n_actions
        hid = max(64, min(320, 800000 // (n_in + spec.n_actions + 320)))
        self.fc1 = nn.Linear(n_in, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.pi = nn.Linear(hid, spec.n_actions)
        self.v = nn.Linear(hid, 1)
        nn.init.zeros_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs, mask):
        x = torch.cat([obs.flatten(1), mask.to(obs.dtype)], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
