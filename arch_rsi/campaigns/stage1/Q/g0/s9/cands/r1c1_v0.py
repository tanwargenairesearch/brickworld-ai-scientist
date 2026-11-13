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
        self.cell = nn.Sequential(nn.Conv2d(WIDTH, HID, 1), nn.ReLU())
        self.type_hidden = nn.ModuleList([nn.Conv2d(2 * HID, HID, 1) for _ in self.footprints])
        self.type_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        g = self.cell(f)
        outs = []
        for (dy, dx), hid, out in zip(self.footprints, self.type_hidden, self.type_out):
            if dy == 1 and dx == 1:
                mx = g
                av = g
            else:
                gp = F.pad(g, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(gp, kernel_size=(dy, dx), stride=1)
                av = F.avg_pool2d(gp, kernel_size=(dy, dx), stride=1)
            h = F.relu(hid(torch.cat([mx, av], dim=1)))
            outs.append(out(h))
        place = torch.cat(outs, dim=1).flatten(1)
        gl = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
