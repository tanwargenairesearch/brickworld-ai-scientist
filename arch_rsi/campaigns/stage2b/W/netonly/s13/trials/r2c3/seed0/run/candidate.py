import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
NBLOCKS = 3


class GBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.gp = nn.Linear(2 * w, w)

    def forward(self, f):
        h = self.conv(f)
        r = F.relu(h)
        b = self.gp(torch.cat([r.mean((2, 3)), r.amax((2, 3))], dim=1))
        return f + F.relu(h + b[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([GBlock(WIDTH) for _ in range(NBLOCKS)])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for blk in self.blocks:
            f = blk(f)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
