import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1}

W = 40
DEPTH = 4
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.stem = nn.Conv2d(spec.in_channels + 2, W, 1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(DEPTH)])
        self.gls = nn.ModuleList([nn.Linear(2 * W, W) for _ in range(DEPTH)])
        self.out = nn.Conv2d(W, self.nt, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = F.relu(self.stem(torch.cat([obs, yy, xx], dim=1)))
        for conv, gl in zip(self.convs, self.gls):
            g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
            h = F.relu(h + conv(h) + gl(g).view(B, W, 1, 1))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
