import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.c1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.broadcast = nn.Linear(WIDTH, WIDTH)
        self.mix = nn.Conv2d(WIDTH, WIDTH, 1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.c1(obs))
        f = F.relu(self.c2(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = f + self.broadcast(g)[:, :, None, None]
        h = F.relu(self.mix(h))
        place = self.place_head(h).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
