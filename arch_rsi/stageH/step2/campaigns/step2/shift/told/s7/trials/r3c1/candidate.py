import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ent_coef": 0.02, "ppo_epochs": 2}

BST = 32
TW = 48
TC = 32
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        X = int(spec.X)
        self.Z = Z
        self.pl = X // 2
        self.pr = X - 1 - X // 2
        self.bstem = nn.Sequential(nn.Conv2d(Z + 2, BST, 1), nn.ReLU())
        self.trow = nn.Conv2d(3 * Z + 1, TW, (1, X))
        self.tproj = nn.Conv2d(TW, TC, 1)
        self.merge = nn.Conv2d(BST + TC, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        bld = torch.cat([obs[:, :Z], obs[:, 4 * Z:4 * Z + 1], obs[:, 4 * Z + 2:4 * Z + 3]], dim=1)
        tgt = torch.cat([obs[:, Z:4 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], dim=1)
        b = self.bstem(bld)
        tp = F.pad(tgt, (self.pl, self.pr, 0, 0), mode='circular')
        t = F.relu(self.tproj(F.relu(self.trow(tp))))
        s = F.relu(self.merge(torch.cat([b, t], dim=1)))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
