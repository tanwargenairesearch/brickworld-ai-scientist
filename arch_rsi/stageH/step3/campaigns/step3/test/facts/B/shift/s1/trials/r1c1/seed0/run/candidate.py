import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
WIDTH = 40
GLOB = 64


def mask_layouts(m, T, Y, X):
    B = m.shape[0]
    a = m.reshape(B, T, Y, X)
    return [a, torch.flip(a, dims=[2]), torch.flip(a, dims=[3]),
            m.reshape(B, T, X, Y).transpose(2, 3)]


def unlayout(maps):
    a = maps[0].flatten(1)
    b = torch.flip(maps[1], dims=[2]).flatten(1)
    c = torch.flip(maps[2], dims=[3]).flatten(1)
    d = maps[3].transpose(2, 3).flatten(1)
    return a + b + c + d


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        cin = spec.in_channels + 4 * self.T + 2
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.ReLU())
        self.c1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.heads = nn.ModuleList([nn.Conv2d(WIDTH, self.T, 1) for _ in range(4)])
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype)
        ms = mask_layouts(m, self.T, self.Y, self.X)
        ys = torch.linspace(-1.0, 1.0, self.Y, device=obs.device, dtype=obs.dtype).view(1, 1, self.Y, 1).expand(B, 1, self.Y, self.X)
        xs = torch.linspace(-1.0, 1.0, self.X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, self.X).expand(B, 1, self.Y, self.X)
        h = torch.cat([obs] + ms + [ys, xs], dim=1)
        s = self.stem(h)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = unlayout([hd(f) for hd in self.heads])
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
