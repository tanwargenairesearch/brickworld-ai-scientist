import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
N_RES = 3


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.norms = nn.ModuleList([nn.GroupNorm(1, WIDTH) for _ in range(N_RES)])
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.out_norm = nn.GroupNorm(1, WIDTH)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.bcast = nn.Linear(WIDTH, WIDTH)
        self.mix = nn.Conv2d(WIDTH, WIDTH, 1)
        self.heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 1, 1) for _ in self.footprints])
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for norm, conv in zip(self.norms, self.convs):
            x = x + conv(F.relu(norm(x)))
        f = F.relu(self.out_norm(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.mix(f) + self.bcast(g)[:, :, None, None])
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, av, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
