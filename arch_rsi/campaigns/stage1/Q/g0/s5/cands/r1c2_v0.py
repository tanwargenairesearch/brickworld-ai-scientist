import math
import torch
import torch.nn as nn

WIDTH = 64
DEPTH = 2


def init_layer(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        g2 = math.sqrt(2.0)
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [init_layer(nn.Conv2d(c_in, WIDTH, 3, padding=1), g2), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.place_head = init_layer(nn.Conv2d(WIDTH, spec.n_types, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, WIDTH), g2), nn.ReLU())
        self.done_head = init_layer(nn.Linear(WIDTH, 1), 0.01)
        self.value_head = init_layer(nn.Linear(WIDTH, 1), 1.0)

    def forward(self, obs, mask):
        f = self.body(obs)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
