import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
W = 40
HH = 32
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
        T = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.Sequential(Block(W), Block(W))
        self.norm = nn.GroupNorm(1, W)
        self.phead = nn.Conv2d(W, HH, 3, padding=1)
        self.pout = nn.Conv2d(HH, T, 1)
        nn.init.normal_(self.pout.weight, std=0.01)
        nn.init.zeros_(self.pout.bias)
        self.lw = nn.Parameter(torch.zeros(4))
        self.done = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))
        nn.init.normal_(self.done[2].weight, std=0.01)
        nn.init.zeros_(self.done[2].bias)
        self.value = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))

    def forward(self, obs, mask):
        f = F.relu(self.norm(self.blocks(self.stem(obs))))
        s = self.pout(F.relu(self.phead(f)))
        c0 = s.flatten(1)
        c1 = s.transpose(2, 3).flatten(1)
        c2 = s.permute(0, 2, 3, 1).flatten(1)
        c3 = s.permute(0, 3, 2, 1).flatten(1)
        w = torch.softmax(self.lw, dim=0)
        place = w[0] * c0 + w[1] * c1 + w[2] * c2 + w[3] * c3
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
