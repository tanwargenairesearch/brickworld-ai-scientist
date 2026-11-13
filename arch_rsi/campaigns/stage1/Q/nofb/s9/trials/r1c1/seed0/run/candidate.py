import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HID = 32


class FootprintHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.convs = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.convs, self.outs):
            h = F.relu(conv(F.pad(f, (0, dx - 1, 0, dy - 1))))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.head = FootprintHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
