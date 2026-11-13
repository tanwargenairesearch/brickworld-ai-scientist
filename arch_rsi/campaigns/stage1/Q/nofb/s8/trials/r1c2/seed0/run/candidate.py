import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
DEPTH = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(DEPTH)])
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        for i, conv in enumerate(self.convs):
            h = h + F.relu(conv(h))
            if i == 0:
                c = self.ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
                h = F.relu(h + c[:, :, None, None])
        h = h / (1.0 + len(self.convs))
        place = self.place_head(h).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
