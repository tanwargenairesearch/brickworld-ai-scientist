"""r1c2 plus a skip connection: the footprint-pooled head's 1x1 projection sees body features and the raw observation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
PW = 32


def fp_pool(p, dy, dx):
    if dy == 1 and dx == 1:
        return torch.cat([p, p, p], dim=1)
    pad = (0, dx - 1, 0, dy - 1)
    mx = F.max_pool2d(F.pad(p, pad, value=-1e4), (dy, dx), stride=1)
    mn = -F.max_pool2d(F.pad(-p, pad, value=-1e4), (dy, dx), stride=1)
    av = F.avg_pool2d(F.pad(p, pad), (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv2d(WIDTH + spec.in_channels, PW, 1)
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PW, PW, 1), nn.ReLU(), nn.Conv2d(PW, 1, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        p = self.proj(torch.cat([f, obs], dim=1))
        outs = []
        for (dy, dx), head in zip(self.fps, self.type_heads):
            outs.append(head(fp_pool(p, dy, dx)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
