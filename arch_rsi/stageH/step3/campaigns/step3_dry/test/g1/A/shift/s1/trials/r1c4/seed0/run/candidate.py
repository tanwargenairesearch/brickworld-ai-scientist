import torch
import torch.nn as nn
import torch.nn.functional as F

W = 48
RED = 16
HIDN = 256


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.stem = nn.Conv2d(C, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.red = nn.Conv2d(W, RED, 1)
        self.fc = nn.Linear(RED * spec.Y * spec.X, HIDN)
        self.mask_fc = nn.Linear(spec.n_actions, HIDN)
        self.fc2 = nn.Linear(HIDN, HIDN)
        self.pi = nn.Linear(HIDN, spec.n_actions)
        nn.init.zeros_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(HIDN, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = F.relu(h + self.c1(h))
        h = F.relu(h + self.c2(h))
        r = F.relu(self.red(h)).flatten(1)
        z = F.relu(self.fc(r) + self.mask_fc(mask.float()))
        z = F.relu(z + self.fc2(z))
        return self.pi(z), self.v(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
