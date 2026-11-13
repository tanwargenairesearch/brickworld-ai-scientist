import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels + 2, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.heads = nn.ModuleList([nn.Conv2d(4 * WIDTH, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype)
        yc = ys.view(1, 1, Y, 1).expand(B, 1, Y, X)
        xc = xs.view(1, 1, 1, X).expand(B, 1, Y, X)
        h = F.relu(self.stem(torch.cat([obs, yc, xc], dim=1)))
        h = h + F.relu(self.c2(h))
        h = h + F.relu(self.c3(h))
        h = h / 3.0
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            fp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(fp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-fp, (dy, dx), stride=1)
            av = F.avg_pool2d(fp, (dy, dx), stride=1)
            outs.append(head(torch.cat([h, mx, mn, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
