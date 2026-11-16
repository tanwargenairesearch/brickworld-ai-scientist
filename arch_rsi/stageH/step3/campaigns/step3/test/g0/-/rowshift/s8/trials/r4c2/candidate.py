import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
W = 48
FC = 16
HID = 256

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        n = int(spec.Y) * int(spec.X)
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(2)])
        self.red = nn.Conv2d(W, FC, 1)
        self.fc = nn.Linear(FC * n, HID)
        self.pol = nn.Linear(HID, int(spec.n_actions))
        nn.init.normal_(self.pol.weight, std=0.01)
        nn.init.zeros_(self.pol.bias)
        self.value_head = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        h = self.stem(torch.cat([obs, m], dim=1))
        for c in self.convs:
            h = F.relu(h + c(h))
        z = F.relu(self.red(h)).flatten(1)
        z = F.relu(self.fc(z))
        logits = self.pol(z)
        return logits, self.value_head(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
