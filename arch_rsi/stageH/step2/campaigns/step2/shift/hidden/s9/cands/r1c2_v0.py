import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
LAND = 16
HID = 32


class FootHead(nn.Module):
    def __init__(self, c, footprints, land, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        n = len(self.fps)
        self.land = nn.ModuleList([nn.Conv2d(c, land, 1) for _ in range(n)])
        self.landp = nn.ModuleList([nn.Conv2d(2 * land, hid, 1) for _ in range(n)])
        self.cellp = nn.ModuleList([nn.Conv2d(c, hid, 1) for _ in range(n)])
        self.mix = nn.ModuleList([nn.Conv2d(3 * hid, hid, 1) for _ in range(n)])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in range(n)])

    def forward(self, f):
        Y = f.shape[2]
        X = f.shape[3]
        res = []
        for k, (dy, dx) in enumerate(self.fps):
            l = self.land[k](f)
            q = F.pad(l, (0, dx - 1, 0, dy - 1))
            lmax = F.max_pool2d(q, (dy, dx), stride=1)
            lavg = F.avg_pool2d(q, (dy, dx), stride=1)
            a = self.landp[k](torch.cat([lmax, lavg], dim=1))
            c = self.cellp[k](f)
            cp = F.pad(c, (0, dx - 1, 0, dy - 1))
            hs = []
            for i in range(dy):
                for j in range(dx):
                    hs.append(F.relu(cp[:, :, i:i + Y, j:j + X] + a))
            h = torch.stack(hs, dim=0)
            z = torch.cat([h.amax(0), h.amin(0), h.mean(0)], dim=1)
            z = F.relu(self.mix[k](z))
            res.append(self.outs[k](z))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = FootHead(WIDTH, spec.footprints, LAND, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
