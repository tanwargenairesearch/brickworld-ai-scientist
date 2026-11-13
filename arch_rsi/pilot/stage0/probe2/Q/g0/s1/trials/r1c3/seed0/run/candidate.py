import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
DEPTH = 4
HID = 16


class FootprintHead(nn.Module):
    def __init__(self, width, footprints, hid):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(width, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, gain=0.01)
            nn.init.zeros_(o.bias)

    def forward(self, f):
        res = []
        for fp, conv, out in zip(self.fps, self.convs, self.outs):
            dy, dx = fp
            h = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(conv(h))))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(DEPTH - 1)])
        self.place_head = FootprintHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for conv in self.blocks:
            f = f + F.relu(conv(f))
        place = self.place_head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
