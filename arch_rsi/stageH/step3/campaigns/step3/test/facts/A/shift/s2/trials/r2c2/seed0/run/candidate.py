import torch
import torch.nn as nn
import torch.nn.functional as F

H1 = 32
H2 = 24
PROJ = 16
HID = 16


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, H1, 1), nn.ReLU(),
            nn.Conv2d(H1, H1, 1), nn.ReLU())
        self.c1 = nn.Conv2d(H1, H2, 3, padding=1)
        self.head = PoolHead(H1 + H2, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * (H1 + H2), 32), nn.ReLU())
        self.done_head = nn.Linear(32, 1)
        self.value_head = nn.Linear(32, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        f = torch.cat([s, F.relu(self.c1(s))], dim=1)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
