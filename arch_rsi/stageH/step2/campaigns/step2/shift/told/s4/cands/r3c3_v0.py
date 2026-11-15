import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.03}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
GD = 64


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * proj, hid, 1) for _ in self.fps])
        self.fp = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, fp, out in zip(self.fps, self.mix, self.fp, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn], dim=1)) + fp(q))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.row = nn.Conv2d(spec.in_channels, STEM, (1, self.X))
        self.stem2 = nn.Conv2d(STEM, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GD), nn.ReLU())
        self.ctx = nn.Linear(GD, WIDTH)
        self.head = FootHead(WIDTH, spec.footprints, PROJ, HID)
        self.done_head = nn.Linear(GD, 1)
        self.value_head = nn.Linear(GD, 1)

    def forward(self, obs, mask):
        X = self.X
        q = F.pad(obs, (X // 2, X - 1 - X // 2, 0, 0), mode='circular')
        s = F.relu(self.stem2(F.relu(self.row(q))))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f2 = F.relu(f + self.ctx(g)[:, :, None, None])
        place = self.head(f2)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
