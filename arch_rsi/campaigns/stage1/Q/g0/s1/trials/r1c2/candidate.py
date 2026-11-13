import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.film = nn.Linear(WIDTH, WIDTH)
        self.mix = nn.Conv2d(WIDTH, WIDTH, 1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f2 = F.relu(self.mix(f) + self.film(g)[:, :, None, None])
        place = self.place_head(f2).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
