import torch
import torch.nn as nn

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "n_minibatches": 16, "ent_coef": 0.003}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W0 = 64
        W = 48
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, W0, 1), nn.ReLU(),
            nn.Conv2d(W0, W0, 1), nn.ReLU(),
            nn.Conv2d(W0, W, 3, padding=1), nn.ReLU(),
            nn.Conv2d(W, W, 3, padding=1), nn.ReLU())
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
