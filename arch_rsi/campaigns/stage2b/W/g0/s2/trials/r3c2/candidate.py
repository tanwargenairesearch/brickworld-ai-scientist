import torch
import torch.nn as nn

COL = 64
W = 64
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, COL, 1), nn.ReLU())
        self.gather = nn.Sequential(nn.Conv2d(COL, W, 3, padding=1), nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Conv2d(W + COL, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.place_head = nn.Conv2d(W + W + COL, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * (W + W + COL), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        c = self.col(obs)
        f1 = torch.cat([self.gather(c), c], dim=1)
        h = self.mlp(f1)
        f = torch.cat([h, f1], dim=1)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
