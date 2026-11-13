"""r2c3 residual+broadcast body with a per-type footprint max/min pooling placement head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

W = 34
HH = 16


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, h):
        return F.relu(h + self.c2(F.relu(self.c1(h))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.block1 = Block(W)
        self.bcast = nn.Linear(2 * W, W)
        self.block2 = Block(W)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * W, HH, 1), nn.ReLU(), nn.Conv2d(HH, 1, 1))
            for _ in self.fps
        ])
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = self.block1(h)
        g0 = self.bcast(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h = F.relu(h + g0[:, :, None, None])
        h = self.block2(h)
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            outs.append(head(torch.cat([h, mx, mn], 1)))
        place = torch.cat(outs, 1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
