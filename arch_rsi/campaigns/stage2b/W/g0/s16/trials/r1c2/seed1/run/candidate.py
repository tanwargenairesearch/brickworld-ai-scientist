import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
N_BLOCKS = 2


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        w = WIDTH
        self.stem = nn.Conv2d(spec.in_channels, w, 3, padding=1)
        self.c1 = nn.ModuleList([nn.Conv2d(w, w, 3, padding=1) for _ in range(N_BLOCKS)])
        self.c2 = nn.ModuleList([nn.Conv2d(w, w, 3, padding=1) for _ in range(N_BLOCKS)])
        self.gl = nn.ModuleList([nn.Linear(2 * w, w) for _ in range(N_BLOCKS)])
        self.place_head = nn.Conv2d(w, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * w, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = F.relu(self.stem(obs))
        for c1, c2, gl in zip(self.c1, self.c2, self.gl):
            g = torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1)
            h = F.relu(c1(x) + gl(g)[:, :, None, None])
            x = F.relu(x + c2(h))
        place = self.place_head(x).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
