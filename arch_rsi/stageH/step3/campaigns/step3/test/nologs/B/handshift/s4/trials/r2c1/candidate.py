import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
W = 48
HID = 256


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(4, c)
        self.c = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cells = spec.Y * spec.X
        cpc = max(4, min(16, 16384 // cells))
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.b1 = Block(W)
        self.b2 = Block(W)
        self.norm = nn.GroupNorm(4, W)
        self.red = nn.Conv2d(W, cpc, 1)
        self.fc = nn.Linear(cpc * cells + 2 * W, HID)
        self.ln = nn.LayerNorm(HID)
        self.fc2 = nn.Linear(HID, HID)
        self.ln2 = nn.LayerNorm(HID)
        self.pi = nn.Linear(HID, spec.n_actions)
        self.v = nn.Linear(HID, 1)
        nn.init.normal_(self.pi.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, obs, mask):
        x = self.stem(obs)
        x = self.b2(self.b1(x))
        f = F.relu(self.norm(x))
        r = F.relu(self.red(f)).flatten(1)
        g = torch.cat([r, f.mean((2, 3)), f.amax((2, 3))], dim=1)
        h = F.relu(self.ln(self.fc(g)))
        h = h + F.relu(self.ln2(self.fc2(h)))
        return self.pi(h), self.v(h).squeeze(1)


def build_policy(spec):
    return Net(spec)
