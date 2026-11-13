import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
RED = 32
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(f[0]), int(f[1])) for f in spec.footprints]
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.reduce = nn.Sequential(nn.Conv2d(WIDTH, RED, 1), nn.ReLU())
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(4 * RED, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        r = self.reduce(f)
        outs = []
        for t, (dy, dx) in enumerate(self.footprints):
            p = F.pad(r, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            h = torch.cat([r, mx, mn, av], dim=1)
            outs.append(self.type_heads[t](h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
