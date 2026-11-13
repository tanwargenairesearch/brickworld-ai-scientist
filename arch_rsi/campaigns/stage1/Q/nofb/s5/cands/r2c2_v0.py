import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
NBLOCK = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels + 2, WIDTH, 3, padding=1)
        self.norms = nn.ModuleList([nn.GroupNorm(1, WIDTH) for _ in range(NBLOCK)])
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(NBLOCK)])
        self.out_norm = nn.GroupNorm(1, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        f = self.stem(torch.cat([obs, ys, xs], dim=1))
        for n, c in zip(self.norms, self.convs):
            f = f + c(F.relu(n(f)))
        f = F.relu(self.out_norm(f))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
