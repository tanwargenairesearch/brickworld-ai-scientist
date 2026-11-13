import math
import torch
import torch.nn as nn
import torch.nn.functional as F

W = 46


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        g2 = math.sqrt(2)
        self.stem = ortho(nn.Conv2d(spec.in_channels, W, 3, padding=1), g2)
        self.c1 = ortho(nn.Conv2d(W, W, 3, padding=1), g2)
        self.ctx = ortho(nn.Linear(2 * W, W), 1.0)
        self.c2 = ortho(nn.Conv2d(W, W, 3, padding=1), g2)
        self.place_head = ortho(nn.Conv2d(W, spec.n_types, 1), 0.01)
        self.glob = ortho(nn.Linear(2 * W, 64), g2)
        self.done_head = ortho(nn.Linear(64, 1), 0.01)
        self.value_head = ortho(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        h0 = F.relu(self.stem(obs))
        g = self.ctx(torch.cat([h0.mean((2, 3)), h0.amax((2, 3))], dim=1))
        h1 = F.relu(self.c1(h0) + g[:, :, None, None])
        h2 = F.relu(self.c2(h1))
        f = h0 + h2
        place = self.place_head(f).flatten(1)
        gg = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
