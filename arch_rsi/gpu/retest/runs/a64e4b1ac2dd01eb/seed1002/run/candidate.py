import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
BLOCKS = 4
PROJ = 16


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.glob = nn.Linear(2 * w, w)

    def forward(self, x):
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        h = self.conv(x) + self.glob(pooled)[:, :, None, None]
        return x + F.relu(h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(BLOCKS)])
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.fconvs = nn.ModuleList([nn.Conv2d(WIDTH, PROJ, (dy, dx)) for (dy, dx) in self.fps])
        self.mixes = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PROJ, PROJ, 1), nn.ReLU(), nn.Conv2d(PROJ, 1, 1))
            for _ in self.fps])
        self.globh = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for b in self.blocks:
            f = b(f)
        p = self.proj(f)
        outs = []
        for (dy, dx), fconv, mix in zip(self.fps, self.fconvs, self.mixes):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            a = fconv(fp)
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            outs.append(mix(torch.cat([a, mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.globh(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
