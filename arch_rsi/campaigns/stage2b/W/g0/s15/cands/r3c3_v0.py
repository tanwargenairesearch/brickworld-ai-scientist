import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.footprints = [(int(a), int(b)) for (a, b) in spec.footprints]
        n = len(self.footprints)
        self.h1 = nn.ModuleList([nn.Conv2d(2 * WIDTH, HID, 1) for _ in range(n)])
        self.ctx = nn.ModuleList([nn.Linear(WIDTH, HID) for _ in range(n)])
        self.h2 = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(n)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        outs = []
        for i, (dy, dx) in enumerate(self.footprints):
            if dy == 1 and dx == 1:
                mx = f
                mn = f
            else:
                p = F.pad(f, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, kernel_size=(dy, dx), stride=1)
                mn = -F.max_pool2d(-p, kernel_size=(dy, dx), stride=1)
            h = self.h1[i](torch.cat([mx, mn], dim=1))
            h = h + self.ctx[i](g).unsqueeze(-1).unsqueeze(-1)
            outs.append(self.h2[i](F.relu(h)))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
