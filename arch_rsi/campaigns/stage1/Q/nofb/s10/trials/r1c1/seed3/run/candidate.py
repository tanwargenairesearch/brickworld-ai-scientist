import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.heads = nn.ModuleList()
        self.outs = nn.ModuleList()
        for (dy, dx) in self.fps:
            self.heads.append(nn.Conv2d(WIDTH, HID, (dy, dx)))
            self.outs.append(nn.Conv2d(HID, 1, 1))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        outs = []
        for (dy, dx), h, o in zip(self.fps, self.heads, self.outs):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(o(F.relu(h(p))))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
