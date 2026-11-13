import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
EMB = 16
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.n_types = spec.n_types
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        T = spec.n_types
        self.cell = nn.Conv2d(WIDTH, T * EMB, 1)
        self.mix = nn.Conv2d(T * 3 * EMB, T * HID, 1, groups=T)
        self.out = nn.Conv2d(T * HID, T, 1, groups=T)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        e = self.cell(f)
        B, _, Y, X = e.shape
        e = e.reshape(B, self.n_types, EMB, Y, X)
        parts = []
        for t, (dy, dx) in enumerate(self.fps):
            et = e[:, t]
            mx = F.max_pool2d(et, (dy, dx), stride=1)
            mn = -F.max_pool2d(-et, (dy, dx), stride=1)
            av = F.avg_pool2d(et, (dy, dx), stride=1)
            p = torch.cat([mx, mn, av], dim=1)
            p = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts.append(p)
        h = torch.cat(parts, dim=1)
        h = F.relu(self.mix(h))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
