import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64


def cln(ln, x):
    return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.c1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.n1 = nn.LayerNorm(WIDTH)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.LayerNorm(WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.LayerNorm(WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f1 = F.relu(cln(self.n1, self.c1(obs)))
        c = self.ctx(torch.cat([f1.mean((2, 3)), f1.amax((2, 3))], dim=1))
        h = self.c2(f1) + c[:, :, None, None]
        f = F.relu(cln(self.n2, h))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
