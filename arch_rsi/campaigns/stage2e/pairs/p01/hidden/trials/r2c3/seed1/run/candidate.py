import torch
import torch.nn as nn
import torch.nn.functional as F

W = 48
RECIPE = {"n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 3, "gamma": 0.99, "gae_lambda": 0.95, "lr": 0.0008}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.cell = nn.Sequential(nn.Conv2d(spec.in_channels, W, 1), nn.ReLU(), nn.Conv2d(W, W, 1), nn.ReLU())
        self.sp1 = nn.Conv2d(W, W, 3, padding=1)
        self.sp2 = nn.Conv2d(W, W, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.film = nn.Linear(W, W)
        self.foot = nn.Conv2d(W, W, 2)
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        nn.init.orthogonal_(self.place_head.weight, 0.01)
        nn.init.zeros_(self.place_head.bias)
        self.done_head = nn.Linear(W, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        c = self.cell(obs)
        h = F.relu(self.sp1(c))
        h = F.relu(self.sp2(h) + c)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h2 = h + self.film(g)[:, :, None, None]
        f = F.relu(self.foot(F.pad(h2, (0, 1, 0, 1))))
        place = self.place_head(f).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
