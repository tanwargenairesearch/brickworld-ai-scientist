import torch
import torch.nn as nn
import torch.nn.functional as F

W = 48
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.c1 = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.gl = nn.Linear(2 * W, W)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        h1 = F.relu(self.c1(obs))
        h2 = F.relu(self.c2(h1)) + h1
        g = F.relu(self.gl(torch.cat([h2.mean((2, 3)), h2.amax((2, 3))], dim=1)))
        h = h2 + g[:, :, None, None]
        h3 = F.relu(self.c3(h)) + h2
        place = self.place_head(h3).flatten(1)
        gg = self.glob(torch.cat([h3.mean((2, 3)), h3.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
