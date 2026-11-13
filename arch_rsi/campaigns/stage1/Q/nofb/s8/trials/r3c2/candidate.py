import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.fconvs = nn.ModuleList([nn.Conv2d(WIDTH, HID, (dy, dx)) for (dy, dx) in self.footprints])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        outs = []
        for (dy, dx), fc, oc in zip(self.footprints, self.fconvs, self.outs):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(oc(F.relu(fc(fp))))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
