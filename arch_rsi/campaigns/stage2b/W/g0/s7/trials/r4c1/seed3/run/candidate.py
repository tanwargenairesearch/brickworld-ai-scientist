"""r3c3 with a nonlinear per-type head on footprint max/min/avg-pooled learned features."""
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2}

WIDTH = 48
QW = 24
HW = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.q = nn.Conv2d(WIDTH, QW, 1)
        self.fps = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        self.pool_hidden = nn.ModuleList([nn.Conv2d(3 * QW, HW, 1) for _ in self.fps])
        self.pool_out = nn.ModuleList([nn.Conv2d(HW, 1, 1) for _ in self.fps])
        for h in list(self.heads) + list(self.pool_out):
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        r = F.relu(self.c1(h))
        h = F.relu(h + self.c2(r))
        q = self.q(h)
        outs = []
        for (dy, dx), head, ph, po in zip(self.fps, self.heads, self.pool_hidden, self.pool_out):
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            qp = F.pad(q, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(qp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-qp, (dy, dx), stride=1)
            av = F.avg_pool2d(qp, (dy, dx), stride=1)
            z = F.relu(ph(torch.cat([mx, mn, av], dim=1)))
            outs.append(head(p) + po(z))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
