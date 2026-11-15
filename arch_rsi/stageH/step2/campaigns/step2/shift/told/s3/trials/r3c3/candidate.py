import torch
import torch.nn as nn
import torch.nn.functional as F

TW = 32
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


def circ_pad_x(t, l, r):
    n = t.shape[-1]
    return torch.cat([t[..., n - l:], t, t[..., :r]], dim=-1)


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.fconv = nn.ModuleList([nn.Conv2d(c, hid, (dy, dx)) for dy, dx in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), fc, mix, out in zip(self.fps, self.fconv, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = F.relu(fc(fp) + mix(torch.cat([mx, mn], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = spec.Z
        self.Z = Z
        self.X = spec.X
        self.cx = nn.Conv2d(3 * Z + 1, TW, (1, spec.X))
        self.fuse1 = nn.Conv2d(TW + Z + 2, COL, 1)
        self.fuse2 = nn.Conv2d(COL, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = FootHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        tgt = torch.cat([obs[:, Z:4 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], dim=1)
        bld = torch.cat([obs[:, :Z], obs[:, 4 * Z:4 * Z + 1], obs[:, 4 * Z + 2:4 * Z + 3]], dim=1)
        lx = self.X // 2
        rx = self.X - 1 - lx
        t = F.relu(self.cx(circ_pad_x(tgt, lx, rx)))
        s = F.relu(self.fuse1(torch.cat([t, bld], dim=1)))
        s = F.relu(self.fuse2(s))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
