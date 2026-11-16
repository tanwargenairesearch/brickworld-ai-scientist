import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.0015, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
WD = 48
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        cin = int(spec.in_channels) + 2
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, WD, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WD, WD, 3, padding=1)
        self.c2 = nn.Conv2d(WD, WD, 3, padding=1)
        self.h1 = nn.Conv2d(WD, WD, 3, padding=1)
        self.h2 = nn.Conv2d(WD, self.T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WD, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = torch.cat([obs, ys, xs], dim=1)
        s = self.stem(h)
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.h2(F.relu(self.h1(f))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
