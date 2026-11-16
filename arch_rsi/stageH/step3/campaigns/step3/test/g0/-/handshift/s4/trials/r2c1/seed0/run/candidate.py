import torch
import torch.nn as nn
import torch.nn.functional as F

CELL = 48
CTX = 24
LAND = 16
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.pad_y = max(a for a, b in self.fps) - 1
        self.pad_x = max(b for a, b in self.fps) - 1
        self.cell = nn.Sequential(
            nn.Conv2d(cin, CELL, 1), nn.ReLU(),
            nn.Conv2d(CELL, CELL, 1), nn.ReLU())
        self.ctx1 = nn.Conv2d(CELL, CTX, 3, padding=1)
        self.ctx2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        n = len(self.fps)
        self.land = nn.ModuleList([nn.Conv2d(CELL, LAND, 1) for _ in range(n)])
        self.cellp = nn.ModuleList([nn.Conv2d(CELL, HID, 1) for _ in range(n)])
        self.landp = nn.ModuleList([nn.Conv2d(LAND, HID, 1, bias=False) for _ in range(n)])
        self.h2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in range(n)])
        self.mix = nn.ModuleList([nn.Conv2d(2 * HID + LAND + CTX, HID, 1) for _ in range(n)])
        self.out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(n)])
        self.glob = nn.Sequential(nn.Linear(2 * (CELL + CTX), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        c = self.cell(obs)
        Y = c.shape[2]
        X = c.shape[3]
        x = F.relu(self.ctx1(c))
        x = F.relu(x + self.ctx2(x))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            lf = F.relu(self.land[i](c))
            if dy * dx > 1:
                L = F.max_pool2d(F.pad(lf, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)
            else:
                L = lf
            a = F.pad(self.cellp[i](c), (0, self.pad_x, 0, self.pad_y))
            lp = self.landp[i](L)
            hs = []
            for oy in range(dy):
                for ox in range(dx):
                    h = F.relu(a[:, :, oy:oy + Y, ox:ox + X] + lp)
                    hs.append(F.relu(self.h2[i](h)))
            hs = torch.stack(hs, 0)
            m = torch.cat([hs.amin(0), hs.mean(0), L, x], dim=1)
            res.append(self.out[i](F.relu(self.mix[i](m))))
        place = torch.cat(res, dim=1).flatten(1)
        f = torch.cat([c, x], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
