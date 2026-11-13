import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HEAD_HID = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 1)
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.heads1 = nn.ModuleList([nn.Conv2d(WIDTH, HEAD_HID, (dy, dx)) for (dy, dx) in self.fps])
        self.heads2 = nn.ModuleList([nn.Conv2d(HEAD_HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        f = F.relu(self.c1(h)) + h
        outs = []
        for (dy, dx), h1, h2 in zip(self.fps, self.heads1, self.heads2):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(h2(F.relu(h1(p))))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
