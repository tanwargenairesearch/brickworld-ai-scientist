import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HW = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv2d(WIDTH, HW, 1), nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * HW, HW, 1), nn.ReLU(), nn.Conv2d(HW, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        p = self.proj(f)
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(q, (dy, dx), stride=1)
            mn = F.avg_pool2d(q, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
