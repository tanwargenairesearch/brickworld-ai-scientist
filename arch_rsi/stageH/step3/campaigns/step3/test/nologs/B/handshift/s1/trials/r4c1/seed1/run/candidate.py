import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
W = 40
K = 16
G = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Y = int(spec.Y)
        X = int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.Sequential(Block(W), Block(W))
        self.norm = nn.GroupNorm(1, W)
        self.red = nn.Conv2d(W, K, 1)
        self.flat = nn.Linear(K * Y * X, int(spec.n_place))
        nn.init.normal_(self.flat.weight, std=0.01)
        nn.init.zeros_(self.flat.bias)
        self.done = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))
        nn.init.normal_(self.done[2].weight, std=0.01)
        nn.init.zeros_(self.done[2].bias)
        self.value = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))

    def forward(self, obs, mask):
        f = F.relu(self.norm(self.blocks(self.stem(obs))))
        place = self.flat(F.relu(self.red(f)).flatten(1))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
