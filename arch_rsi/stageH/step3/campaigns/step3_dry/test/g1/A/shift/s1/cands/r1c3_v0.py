import torch
import torch.nn as nn
import torch.nn.functional as F

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
        for o in self.outs:
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)

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
        self.stem = nn.Sequential(
            nn.Conv2d(C, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.done_feat = nn.Conv2d(C, 32, 1)
        self.done_mlp = nn.Sequential(nn.Linear(64 + 2 * WIDTH, 32), nn.ReLU())
        self.done_out = nn.Linear(32, 1)
        nn.init.zeros_(self.done_out.weight)
        nn.init.zeros_(self.done_out.bias)
        self.v_stem = nn.Conv2d(C, 32, 1)
        self.v_conv = nn.Conv2d(32, 32, 3, padding=1)
        self.v_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        d = F.relu(self.done_feat(obs))
        dg = torch.cat([d.mean((2, 3)), d.amax((2, 3)), f.mean((2, 3)), f.amax((2, 3))], dim=1)
        done = self.done_out(self.done_mlp(dg))
        v = F.relu(self.v_stem(obs))
        v = F.relu(v + self.v_conv(v))
        value = self.v_mlp(torch.cat([v.mean((2, 3)), v.amax((2, 3))], dim=1)).squeeze(1)
        logits = torch.cat([place, done], dim=1)
        return logits, value


def build_policy(spec):
    return Net(spec)
