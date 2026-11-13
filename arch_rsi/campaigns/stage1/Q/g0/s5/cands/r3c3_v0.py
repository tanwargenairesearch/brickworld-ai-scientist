import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
BLOCKS = 3


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.glob = nn.Linear(2 * w, w)

    def forward(self, x):
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
        h = self.conv(x) + self.glob(pooled)[:, :, None, None]
        return x + F.relu(h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(BLOCKS)])
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.place_heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for (dy, dx) in self.fps])
        self.globh = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for b in self.blocks:
            f = b(f)
        outs = []
        for (dy, dx), head in zip(self.fps, self.place_heads):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(head(fp))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.globh(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
