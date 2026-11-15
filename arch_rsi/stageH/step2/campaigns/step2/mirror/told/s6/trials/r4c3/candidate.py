import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'ppo_epochs': 2, 'ent_coef': 0.02, 'n_minibatches': 16, 'lr': 0.002}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
GDIM = 64


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid, gdim):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.gproj = nn.Linear(gdim, proj)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f, g):
        p = self.proj(f) + self.gproj(g).unsqueeze(2).unsqueeze(3)
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
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(2 * WIDTH, spec.footprints, PROJ, HID, GDIM)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GDIM), nn.ReLU())
        self.done_head = nn.Linear(GDIM, 1)
        self.value_head = nn.Linear(GDIM, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        fm = torch.flip(f, dims=[3])
        place = self.head(torch.cat([f, fm], dim=1), g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
