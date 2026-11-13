import torch
import torch.nn as nn
import torch.nn.functional as F

COLH = 64
WIDTH = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.col1 = nn.Conv2d(spec.in_channels, COLH, 1)
        self.col2 = nn.Conv2d(COLH, WIDTH, 1)
        self.conv1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.bcast = nn.Linear(2 * WIDTH, WIDTH)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        c = F.relu(self.col1(obs))
        c = F.relu(self.col2(c))
        f = F.relu(c + self.conv1(c))
        g0 = self.bcast(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = F.relu(f + g0[:, :, None, None])
        f = F.relu(f + self.conv2(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
