import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005, "n_envs": 128, "rollout_len": 32, "n_minibatches": 64}

STEM = 64
W = 48
HH = 32
G = 64
NB = 3


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
        self.T = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.Sequential(*[Block(W) for _ in range(NB)])
        self.norm = nn.GroupNorm(1, W)
        self.phead = nn.Conv2d(W, HH, 3, padding=1)
        self.pout = nn.Conv2d(HH, self.T, 1)
        nn.init.normal_(self.pout.weight, std=0.01)
        nn.init.zeros_(self.pout.bias)
        self.done = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))
        nn.init.zeros_(self.done[2].weight)
        nn.init.zeros_(self.done[2].bias)
        self.value = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, 1))

    def forward(self, obs, mask):
        f = F.relu(self.norm(self.blocks(self.stem(obs))))
        place = self.pout(F.relu(self.phead(f))).flatten(1)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        d = self.done(g)
        done_logit = torch.logsumexp(place, dim=1, keepdim=True) + d
        logits = torch.cat([place, done_logit], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
