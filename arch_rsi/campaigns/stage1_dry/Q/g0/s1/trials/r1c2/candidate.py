import torch
import torch.nn as nn
import torch.nn.functional as F

W = 40
WH = 24
WG = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(f[0]), int(f[1])) for f in spec.footprints]
        self.c1 = nn.Conv2d(spec.in_channels, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.heads = nn.ModuleList()
        for _ in self.fps:
            last = nn.Conv2d(WH, 1, 1)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.heads.append(nn.Sequential(nn.Conv2d(3 * W, WH, 1), nn.ReLU(), last))
        self.glob = nn.Sequential(nn.Linear(2 * W, WG), nn.ReLU())
        self.done_head = nn.Linear(WG, 1)
        self.value_head = nn.Linear(WG, 1)

    def forward(self, obs, mask):
        h = F.relu(self.c1(obs))
        h = h + F.relu(self.c2(h))
        h = h + F.relu(self.c3(h))
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            outs.append(head(torch.cat([h, mx, mn], dim=1)).squeeze(1))
        place = torch.stack(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
