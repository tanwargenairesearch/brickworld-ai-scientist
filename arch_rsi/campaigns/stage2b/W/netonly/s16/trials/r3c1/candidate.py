import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 30
NBLK = 5
HID = 20


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLK)])
        self.proj = nn.Conv2d(WIDTH, HID, 1)
        self.type_hidden = nn.ModuleList([nn.Conv2d(4 * HID, HID, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for conv in self.blocks:
            x = x + F.relu(conv(x))
        h = self.proj(x)
        outs = []
        for i, fp in enumerate(self.fps):
            dy, dx = fp
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            z = torch.cat([h, mx, mn, av], dim=1)
            z = F.relu(self.type_hidden[i](z))
            outs.append(self.type_out[i](z))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
