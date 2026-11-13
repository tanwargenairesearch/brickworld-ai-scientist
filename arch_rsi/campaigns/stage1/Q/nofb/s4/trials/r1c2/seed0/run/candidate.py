import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
BLOCKS = 3


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(BLOCKS)])
        self.bcast = nn.Linear(2 * WIDTH, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for i, blk in enumerate(self.blocks):
            f = blk(f)
            if i == 0:
                g0 = self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
                f = F.relu(f + g0[:, :, None, None])
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
