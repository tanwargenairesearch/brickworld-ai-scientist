"""Per-cell stem with mask input, three residual 3x3 blocks with global broadcasts, 2x2 footprint head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
FOOT = 32
N_BLOCKS = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        W = WIDTH
        self.n_types = spec.n_types
        self.n_place = spec.n_place
        self.Y = spec.Y
        self.X = spec.X
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + spec.n_types, W, 1), nn.ReLU(),
            nn.Conv2d(W, W, 1), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(N_BLOCKS)])
        self.bcasts = nn.ModuleList([nn.Linear(2 * W, W) for _ in range(N_BLOCKS - 1)])
        self.foot = nn.Conv2d(W, FOOT, 2)
        self.place_head = nn.Conv2d(FOOT + W, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.n_types, self.Y, self.X)
        f = self.stem(torch.cat([obs, m], dim=1))
        for i, conv in enumerate(self.blocks):
            f = f + F.relu(conv(f))
            if i < len(self.bcasts):
                g0 = F.relu(self.bcasts[i](torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)))
                f = f + g0[:, :, None, None]
        p = F.relu(self.foot(F.pad(f, (0, 1, 0, 1))))
        place = self.place_head(torch.cat([p, f], dim=1)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
