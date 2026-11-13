import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
PROJ = 32
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        T = spec.n_types
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.skip = nn.Conv2d(spec.in_channels, PROJ, 1)
        self.type_hidden = nn.Conv2d(T * 3 * PROJ, T * HID, 1, groups=T)
        self.type_out = nn.Conv2d(T * HID, T, 1, groups=T)
        nn.init.zeros_(self.type_out.weight)
        nn.init.zeros_(self.type_out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * PROJ, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        p = F.relu(self.proj(f) + self.skip(obs))
        pooled = []
        for (dy, dx) in self.footprints:
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            av = F.avg_pool2d(q, (dy, dx), stride=1)
            pooled += [mx, mn, av]
        h = torch.cat(pooled, dim=1)
        place = self.type_out(F.relu(self.type_hidden(h))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), p.mean((2, 3)), p.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
