import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 33
BLOCKS = 5


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.place_heads = nn.ModuleList([nn.Conv2d(WIDTH, 1, (dy, dx)) for dy, dx in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for conv in self.blocks:
            x = x + conv(F.relu(x))
        f = F.relu(x)
        outs = []
        for (dy, dx), head in zip(self.fps, self.place_heads):
            outs.append(head(F.pad(f, (0, dx - 1, 0, dy - 1))))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
