import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 46
HEAD_CH = 8


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.n_types = spec.n_types
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.cell_head = nn.Conv2d(WIDTH, spec.n_types * HEAD_CH, 1)
        self.combine = nn.Conv2d(spec.n_types * 3 * HEAD_CH, spec.n_types, 1, groups=spec.n_types)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        h = self.cell_head(f)
        outs = []
        for t, (dy, dx) in enumerate(self.footprints):
            ht = h[:, t * HEAD_CH:(t + 1) * HEAD_CH]
            hp = F.pad(ht, (0, dx - 1, 0, dy - 1))
            mean = F.avg_pool2d(hp, (dy, dx), stride=1)
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            outs += [mean, mn, mx]
        place = self.combine(torch.cat(outs, dim=1)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
