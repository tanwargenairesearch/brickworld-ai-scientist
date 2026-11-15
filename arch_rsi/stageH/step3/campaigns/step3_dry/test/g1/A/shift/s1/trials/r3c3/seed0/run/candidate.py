import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "n_minibatches": 16, "gamma": 0.99, "gae_lambda": 0.95}

W = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = spec.n_types
        self.Y = spec.Y
        self.X = spec.X
        self.n_place = spec.n_place
        self.stem = nn.Conv2d(spec.in_channels + spec.n_types, W, 1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(3)])
        self.out = nn.Conv2d(W, spec.n_types, 5, padding=2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done = nn.Linear(64, 1)
        nn.init.zeros_(self.done.weight)
        nn.init.zeros_(self.done.bias)
        self.value = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].float().reshape(B, self.T, self.Y, self.X)
        h = F.relu(self.stem(torch.cat([obs, m], dim=1)))
        for c in self.convs:
            h = F.relu(h + c(h))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
