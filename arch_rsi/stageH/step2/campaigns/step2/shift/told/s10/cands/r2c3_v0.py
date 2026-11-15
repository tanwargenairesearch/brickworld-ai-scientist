import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {'n_envs': 64, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 4}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32


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


class AxisPerm(nn.Module):
    def __init__(self, c, ny, nx):
        super().__init__()
        self.wx = nn.Parameter(torch.empty(c, nx, nx))
        self.wy = nn.Parameter(torch.empty(c, ny, ny))
        nn.init.uniform_(self.wx, -math.sqrt(3.0 / nx), math.sqrt(3.0 / nx))
        nn.init.uniform_(self.wy, -math.sqrt(3.0 / ny), math.sqrt(3.0 / ny))

    def forward(self, x):
        a = torch.einsum('bcyx,cxw->bcyw', x, self.wx)
        a = torch.einsum('bcyw,cyv->bcvw', a, self.wy)
        return a


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.perm = AxisPerm(cin, int(spec.Y), int(spec.X))
        self.stem = nn.Sequential(
            nn.Conv2d(2 * cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        a = self.perm(obs)
        s = self.stem(torch.cat([obs, a], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
