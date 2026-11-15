import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {'ppo_epochs': 1, 'n_minibatches': 64}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


def xroll(t, s):
    return torch.cat([t[..., s:], t[..., :s]], dim=-1)


class AxialMix(nn.Module):
    def __init__(self, c, ny, nx):
        super().__init__()
        self.wx = nn.Parameter(torch.empty(c, nx, nx))
        self.wy = nn.Parameter(torch.empty(c, ny, ny))
        self.bx = nn.Parameter(torch.zeros(c, 1, nx))
        self.by = nn.Parameter(torch.zeros(c, ny, 1))
        nn.init.uniform_(self.wx, -1.0 / math.sqrt(nx), 1.0 / math.sqrt(nx))
        nn.init.uniform_(self.wy, -1.0 / math.sqrt(ny), 1.0 / math.sqrt(ny))

    def forward(self, f):
        mx = torch.einsum('bcyx,cxw->bcyw', f, self.wx) + self.bx
        my = torch.einsum('bcyx,cyv->bcvx', f, self.wy) + self.by
        return mx, my


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
        C = spec.in_channels
        self.Z = spec.Z
        self.sh = spec.X // 2
        nroll = 3 * spec.Z + 1
        self.inmix = AxialMix(C, spec.Y, spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(3 * C + nroll, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.mix2 = AxialMix(WIDTH, spec.Y, spec.X)
        self.fuse = nn.Conv2d(3 * WIDTH, WIDTH, 1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        tsel = torch.cat([obs[:, Z:4 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], dim=1)
        rolled = xroll(tsel, self.sh)
        mx, my = self.inmix(obs)
        s = self.stem(torch.cat([obs, rolled, mx, my], dim=1))
        x = F.relu(self.c1(s))
        ax, ay = self.mix2(x)
        x = F.relu(x + self.fuse(torch.cat([x, ax, ay], dim=1)))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
