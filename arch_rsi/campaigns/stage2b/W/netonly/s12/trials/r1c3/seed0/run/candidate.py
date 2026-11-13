import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 38
N_BLOCKS = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Conv2d(c, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_BLOCKS)])
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        for i, conv in enumerate(self.blocks):
            if i == 1:
                g0 = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
                f = f + F.relu(self.ctx(g0))[:, :, None, None]
            f = f + F.relu(conv(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
