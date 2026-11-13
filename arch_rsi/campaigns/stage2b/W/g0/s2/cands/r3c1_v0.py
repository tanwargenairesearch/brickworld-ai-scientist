import torch
import torch.nn as nn

COL = 64
WIDTH = 48
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, COL, 1), nn.ReLU())
        self.body = nn.Sequential(
            nn.Conv2d(COL, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.place_head = nn.Conv2d(WIDTH + COL, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * (WIDTH + COL), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        c = self.col(obs)
        f = torch.cat([self.body(c), c], dim=1)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
