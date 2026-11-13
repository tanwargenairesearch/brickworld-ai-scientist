import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2


class ConvLN(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.ln = nn.LayerNorm(c_out)

    def forward(self, x):
        h = self.conv(x)
        h = self.ln(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return F.relu(h)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers.append(ConvLN(c_in, WIDTH))
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.LayerNorm(WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
