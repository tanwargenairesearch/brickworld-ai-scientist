import torch
import torch.nn as nn
import torch.nn.functional as F

W = 48
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.head = nn.Conv2d(W, spec.n_types, 2)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.head.weight, 0.01)
        nn.init.zeros_(self.head.bias)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = F.relu(self.c1(h)) + h
        h = F.relu(self.c2(h)) + h
        place = self.head(F.pad(h, (0, 1, 0, 1))).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
