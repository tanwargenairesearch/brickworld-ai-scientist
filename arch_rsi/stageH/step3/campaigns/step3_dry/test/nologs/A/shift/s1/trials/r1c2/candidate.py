import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1}

STEM = 64
COL = 48
WIDTH = 40
HID = 32
PRJ = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.skip = nn.Conv2d(COL, WIDTH, 1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.proj = nn.Conv2d(WIDTH, PRJ, 1)
        self.fconv = nn.ModuleList([nn.Conv2d(WIDTH, HID, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID + PRJ, 1, 1) for _ in self.fps])
        with torch.no_grad():
            for o in self.outs:
                o.weight.mul_(0.1)
                o.bias.zero_()
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s) + self.skip(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(f)
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.fconv, self.outs):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            h = conv(q)
            if dy * dx == 1:
                m = p
            else:
                m = F.max_pool2d(F.pad(p, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)
            res.append(out(F.relu(torch.cat([h, m], dim=1))))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
