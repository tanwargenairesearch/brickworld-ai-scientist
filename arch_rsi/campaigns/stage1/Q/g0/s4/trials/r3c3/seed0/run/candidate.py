import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        g2 = math.sqrt(2.0)
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [init_layer(nn.Conv2d(c_in, WIDTH, 3, padding=1), g2), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.fps = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.head_hidden = init_layer(nn.Conv2d(WIDTH, WIDTH, 1), g2)
        self.type_heads = nn.ModuleList([init_layer(nn.Conv2d(3 * WIDTH, 1, 1), 0.01) for _ in self.fps])
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, WIDTH), g2), nn.ReLU())
        self.done_head = init_layer(nn.Linear(WIDTH, 1), 0.01)
        self.value_head = init_layer(nn.Linear(WIDTH, 1), 1.0)

    def forward(self, obs, mask):
        f = self.body(obs)
        h = F.relu(self.head_hidden(f))
        outs = []
        for (dy, dx), head in zip(self.fps, self.type_heads):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, av, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
