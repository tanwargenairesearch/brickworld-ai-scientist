import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.conv1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.inject = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH))
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.conv1(obs))
        c = self.inject(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = f + c[:, :, None, None]
        f = F.relu(self.conv2(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
