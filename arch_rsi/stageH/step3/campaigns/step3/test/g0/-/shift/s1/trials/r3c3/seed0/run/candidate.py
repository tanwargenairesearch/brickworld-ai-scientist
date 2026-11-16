import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 4, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "lr": 0.0005, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01}

STEM = 64
W = 48
R = 8
G = 64
DEPTH = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        cells = self.Y * self.X
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(DEPTH)])
        self.local = nn.Conv2d(W, self.T, 3, padding=1)
        self.red = nn.Conv2d(W, R, 1)
        self.dense = nn.Linear(R * cells, int(spec.n_place))
        self.glob = nn.Sequential(nn.Linear(2 * W + R * cells, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat([obs, ys, xs], dim=1))
        for conv in self.convs:
            h = F.relu(h + conv(h))
        flat = F.relu(self.red(h)).flatten(1)
        place = self.local(h).flatten(1) + self.dense(flat)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), flat], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
