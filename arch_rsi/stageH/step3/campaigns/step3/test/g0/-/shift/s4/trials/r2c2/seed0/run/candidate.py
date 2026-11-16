import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95}

W = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(3)])
        self.gproj = nn.Linear(2 * W, W)
        self.hmix = nn.Conv2d(2 * W, W, 1)
        self.out = nn.Conv2d(W, self.nt, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat([obs, ys, xs], dim=1))
        for c in self.convs:
            h = F.relu(h + c(h))
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        gb = self.gproj(pooled).view(B, W, 1, 1).expand(B, W, Y, X)
        z = F.relu(self.hmix(torch.cat([h, gb], dim=1)))
        place = self.out(z).flatten(1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
