import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 41
BLOCKS = 3
K = 16
H = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.lin_heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        self.cell = nn.Conv2d(WIDTH, K, 1)
        self.pool_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * K, H, 1), nn.ReLU(), nn.Conv2d(H, 1, 1)) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.vglob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for conv in self.blocks:
            x = x + conv(F.relu(x))
        f = F.relu(x)
        u = self.cell(f)
        outs = []
        for (dy, dx), lin, ph in zip(self.fps, self.lin_heads, self.pool_heads):
            pad = (0, dx - 1, 0, dy - 1)
            fp_ = F.pad(f, pad)
            up = F.pad(u, pad)
            mx = F.max_pool2d(up, (dy, dx), stride=1)
            mn = -F.max_pool2d(-up, (dy, dx), stride=1)
            av = F.avg_pool2d(up, (dy, dx), stride=1)
            outs.append(lin(fp_) + ph(torch.cat([mx, mn, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        value = self.value_head(self.vglob(pooled.detach())).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
