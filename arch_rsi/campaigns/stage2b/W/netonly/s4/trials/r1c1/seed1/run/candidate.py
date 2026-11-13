import math
import torch
import torch.nn as nn

WIDTH = 64
DEPTH = 2


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [ortho(nn.Conv2d(c_in, WIDTH, 3, padding=1), math.sqrt(2)), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.place_head = ortho(nn.Conv2d(WIDTH, spec.n_types, 1), 0.01)
        self.glob = nn.Sequential(ortho(nn.Linear(2 * WIDTH, WIDTH), math.sqrt(2)), nn.ReLU())
        self.done_head = ortho(nn.Linear(WIDTH, 1), 0.01)
        self.value_head = ortho(nn.Linear(WIDTH, 1), 1.0)

    def forward(self, obs, mask):
        f = self.body(obs)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
