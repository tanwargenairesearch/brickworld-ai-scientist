import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
HID = 32


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain=gain)
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
        self.footprints = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.heads = nn.ModuleList([
            nn.Sequential(init_layer(nn.Conv2d(2 * WIDTH, HID, 1), g2), nn.ReLU(),
                          init_layer(nn.Conv2d(HID, 1, 1), 0.01))
            for _ in self.footprints
        ])
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH, WIDTH), g2), nn.ReLU())
        self.done_head = init_layer(nn.Linear(WIDTH, 1), 0.01)
        self.value_head = init_layer(nn.Linear(WIDTH, 1), 1.0)

    def forward(self, obs, mask):
        f = self.body(obs)
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            if dy == 1 and dx == 1:
                mx = f
                mn = f
            else:
                p = F.pad(f, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, kernel_size=(dy, dx), stride=1)
                mn = -F.max_pool2d(-p, kernel_size=(dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
