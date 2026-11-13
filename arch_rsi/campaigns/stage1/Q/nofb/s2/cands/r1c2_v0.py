import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
BLOCKS = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.globs = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(BLOCKS)])
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        for conv, gl in zip(self.convs, self.globs):
            g = gl(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
            h = h + F.relu(conv(F.relu(h + g[:, :, None, None])))
        place = self.place_head(h).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
