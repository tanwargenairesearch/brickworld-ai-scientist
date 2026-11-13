import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
BLOCKS = 2
LW = 8


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.footprints = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.layer_conv = nn.Sequential(
            nn.Conv2d(7, LW, 3, padding=1), nn.ReLU(),
            nn.Conv2d(LW, LW, 1), nn.ReLU())
        self.sel_heads = nn.ModuleList([nn.Conv2d(4 * WIDTH, spec.Z, 1) for _ in self.footprints])
        self.type_heads = nn.ModuleList([nn.Conv2d(4 * WIDTH + 2 * LW, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        f = self.blocks(self.stem(obs))
        b = obs[:, 0:Z]
        t_ = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0))[:, :Z]
        tb = F.pad(t_, (0, 0, 0, 0, 1, 0))[:, :Z]
        zpos = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, Z, 1, 1).expand(B, Z, Y, X)
        per = torch.stack([b, t_, m, e, bb, tb, zpos], dim=2).reshape(B * Z, 7, Y, X)
        L = self.layer_conv(per)
        outs = []
        for k, (dy, dx) in enumerate(self.footprints):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            h = torch.cat([f, mx, mn, av], dim=1)
            lp = F.pad(L, (0, dx - 1, 0, dy - 1))
            lmx = F.max_pool2d(lp, (dy, dx), stride=1)
            lmn = -F.max_pool2d(-lp, (dy, dx), stride=1)
            lf = torch.cat([lmx, lmn], dim=1).reshape(B, Z, 2 * LW, Y, X)
            q = torch.softmax(self.sel_heads[k](h), dim=1)
            sel = (q.unsqueeze(2) * lf).sum(1)
            outs.append(self.type_heads[k](torch.cat([h, sel], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
