import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 38
BLOCKS = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.mid_glob = nn.Linear(2 * WIDTH, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for i, conv in enumerate(self.blocks):
            x = x + conv(F.relu(x))
            if i == 0:
                r = F.relu(x)
                g0 = self.mid_glob(torch.cat([r.mean((2, 3)), r.amax((2, 3))], dim=1))
                x = x + g0[:, :, None, None]
        f = F.relu(x)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
