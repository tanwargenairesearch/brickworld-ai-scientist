import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "ppo_epochs": 1, "n_minibatches": 8, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.01}

STEM = 64
W = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.square = int(spec.Y) == int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.h1 = nn.Conv2d(W, W, 3, padding=1)
        self.h2 = nn.Conv2d(W, self.nt, 1)
        self.h2t = nn.Conv2d(W, self.nt, 1)
        nn.init.normal_(self.h2.weight, std=0.01)
        nn.init.zeros_(self.h2.bias)
        nn.init.normal_(self.h2t.weight, std=0.01)
        nn.init.zeros_(self.h2t.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def add_coords(self, obs):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        return torch.cat([obs, ys, xs], dim=1)

    def forward(self, obs, mask):
        s = self.stem(self.add_coords(obs))
        x = F.relu(s + self.c1(s))
        f = F.relu(x + self.c2(x))
        h = F.relu(self.h1(f))
        sc = self.h2(h)
        if self.square:
            sc = sc + self.h2t(h).transpose(2, 3)
        place = sc.flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
