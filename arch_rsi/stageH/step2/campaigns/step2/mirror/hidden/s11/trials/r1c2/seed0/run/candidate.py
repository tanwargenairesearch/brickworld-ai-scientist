import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3, "gamma": 0.97}

S1 = 96
COL = 48
WIDTH = 48
PROJ = 32
HID = 32
GD = 64


class Head(nn.Module):
    def __init__(self, c, footprints, proj, hid, gdim):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.gb = nn.ModuleList([nn.Linear(gdim, hid) for _ in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f, g):
        p = self.proj(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = self.mix[i](torch.cat([mx, mn, av], dim=1))
            h = F.relu(h + self.gb[i](g)[:, :, None, None])
            h = F.relu(h + self.mix2[i](h))
            res.append(self.outs[i](h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.s1 = nn.Conv2d(spec.in_channels, S1, 1)
        self.s2 = nn.Conv2d(S1, COL, 1)
        self.s3 = nn.Conv2d(COL, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GD), nn.ReLU())
        self.head = Head(WIDTH, spec.footprints, PROJ, HID, GD)
        self.done_head = nn.Linear(GD, 1)
        self.value_head = nn.Linear(GD, 1)

    def forward(self, obs, mask):
        s = F.relu(self.s2(F.relu(self.s1(obs))))
        s = F.relu(s + self.s3(s))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(f, g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
