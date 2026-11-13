import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
NBLOCK = 2
HID = 32


class FootHead(nn.Module):
    def __init__(self, c, footprints, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for (a, b) in footprints]
        self.pre = nn.Conv2d(c, hid, 1)
        self.mlps = nn.ModuleList()
        for _ in self.fps:
            out = nn.Conv2d(hid, 1, 1)
            nn.init.normal_(out.weight, std=0.01)
            nn.init.zeros_(out.bias)
            self.mlps.append(nn.Sequential(nn.Conv2d(3 * hid, hid, 1), nn.ReLU(), out))

    def forward(self, f):
        h = self.pre(f)
        outs = []
        for fp, mlp in zip(self.fps, self.mlps):
            dy, dx = fp
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            outs.append(mlp(torch.cat([mx, mn, av], 1)))
        return torch.cat(outs, 1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.inp = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.ctxs = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(NBLOCK + 1)])
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLOCK)])
        self.place_head = FootHead(WIDTH, spec.footprints, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.inp(obs))
        for i in range(NBLOCK + 1):
            c = F.relu(self.ctxs[i](torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
            f = f + c.unsqueeze(2).unsqueeze(3)
            if i < NBLOCK:
                f = f + F.relu(self.blocks[i](f))
        place = self.place_head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
