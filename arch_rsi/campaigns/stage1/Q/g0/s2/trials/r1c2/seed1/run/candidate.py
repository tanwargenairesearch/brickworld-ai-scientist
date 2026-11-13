import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 40
N_RES = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.mid_glob = nn.Linear(2 * WIDTH, WIDTH)
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        x = F.relu(self.stem(obs))
        for i, conv in enumerate(self.res):
            x = x + F.relu(conv(x))
            if i == 0:
                g = self.mid_glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
                x = F.relu(x + g[:, :, None, None])
        place = self.place_head(x).flatten(1)
        g = self.glob(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
