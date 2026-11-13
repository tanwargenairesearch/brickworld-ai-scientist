import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
BLOCKS = 2


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.type_heads = nn.ModuleList([nn.Conv2d(4 * WIDTH, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.film = nn.Linear(64, 4 * WIDTH)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.blocks(self.stem(obs))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        gamma = self.film(g)[:, :, None, None]
        outs = []
        for t, (dy, dx) in enumerate(self.footprints):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            h = torch.cat([f, mx, mn, av], dim=1)
            h = h * (1.0 + gamma)
            outs.append(self.type_heads[t](h))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
