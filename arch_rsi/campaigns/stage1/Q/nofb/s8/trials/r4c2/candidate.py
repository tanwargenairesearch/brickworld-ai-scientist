import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
RED = 32
HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.reduce = nn.Conv2d(WIDTH, RED, 1)
        self.hid = nn.ModuleList([nn.Conv2d(4 * RED, HID, 1) for _ in self.footprints])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        r = F.relu(self.reduce(f))
        outs = []
        for (dy, dx), hc, oc in zip(self.footprints, self.hid, self.outs):
            fp = F.pad(r, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(fp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-fp, (dy, dx), stride=1)
            av = F.avg_pool2d(fp, (dy, dx), stride=1)
            z = F.relu(hc(torch.cat([r, mx, mn, av], dim=1)))
            outs.append(oc(z))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
