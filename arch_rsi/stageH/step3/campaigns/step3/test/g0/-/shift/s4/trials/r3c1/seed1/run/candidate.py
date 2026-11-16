import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.001, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95}

W = 48
R = 8
G = 128


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1), nn.ReLU())
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(2)])
        self.red = nn.Conv2d(W, R, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W + int(spec.n_actions), G), nn.ReLU())
        self.act = nn.Linear(R * n + G, int(spec.n_actions))
        nn.init.orthogonal_(self.act.weight, 0.01)
        nn.init.zeros_(self.act.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat([obs, ys, xs], dim=1))
        for c in self.convs:
            h = F.relu(h + c(h))
        r = F.relu(self.red(h)).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), mask.to(obs.dtype)], dim=1))
        logits = self.act(torch.cat([r, g], dim=1))
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
