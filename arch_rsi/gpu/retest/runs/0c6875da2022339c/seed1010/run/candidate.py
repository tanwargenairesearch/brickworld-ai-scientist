"""Stage 1 comparator, no search: the starting architecture plus two standard PPO practices, written before any
Stage 1 result. (1) A separate critic: the value head gets its own conv body and pooled layer, so the value loss
no longer shapes the policy's features. (2) Orthogonal initialisation: gain sqrt(2) for hidden layers, 0.01 for
the policy outputs (placement and DONE), 1 for the value output, zero biases. Everything else is the start's."""
import math

import torch
import torch.nn as nn

WIDTH = 64
DEPTH = 2


def conv_body(c_in):
    layers = []
    for _ in range(DEPTH):
        layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
        c_in = WIDTH
    return nn.Sequential(*layers)


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.body = conv_body(spec.in_channels)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.critic_body = conv_body(spec.in_channels)
        self.critic_glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.value_head = nn.Linear(WIDTH, 1)
        hidden = list(self.body) + list(self.critic_body) + [self.glob[0], self.critic_glob[0]]
        for m in hidden:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                ortho(m, math.sqrt(2))
        ortho(self.place_head, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        f = self.body(obs)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = self.critic_body(obs)
        v = self.critic_glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        return torch.cat([place, self.done_head(g)], dim=1), self.value_head(v).squeeze(1)


def build_policy(spec):
    return Net(spec)
