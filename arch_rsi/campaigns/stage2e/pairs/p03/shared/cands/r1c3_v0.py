import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
BLOCKS = 3


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.ctx = nn.Linear(2 * w, w)

    def forward(self, h):
        g = self.ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        return h + F.relu(self.conv(h) + g[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.place_head = nn.Sequential(
            nn.Conv2d(WIDTH, WIDTH, 1), nn.ReLU(), nn.Conv2d(WIDTH, spec.n_types, 1))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.blocks(self.stem(obs))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
