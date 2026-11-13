import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
DEPTH = 4
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(DEPTH - 1)])
        self.footprints = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.footprints
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for conv in self.blocks:
            f = f + F.relu(conv(f))
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            if dy == 1 and dx == 1:
                mx = f
                mn = f
            else:
                p = F.pad(f, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, kernel_size=(dy, dx), stride=1)
                mn = -F.max_pool2d(-p, kernel_size=(dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
