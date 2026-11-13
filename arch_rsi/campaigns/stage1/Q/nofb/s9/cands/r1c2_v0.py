import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
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


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.b1 = ResBlock(WIDTH)
        self.mid_glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH))
        self.b2 = ResBlock(WIDTH)
        self.head = FootprintHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = self.b1(self.stem(obs))
        r = F.relu(x)
        gm = self.mid_glob(torch.cat([r.mean((2, 3)), r.amax((2, 3))], dim=1))
        x = x + gm[:, :, None, None]
        f = F.relu(self.b2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
