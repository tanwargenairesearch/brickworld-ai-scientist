import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
DEPTH = 2
PROJ = 32
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Conv2d(WIDTH, PROJ, 1), nn.ReLU())
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        heads = []
        for _ in self.fps:
            h = nn.Sequential(nn.Conv2d(3 * PROJ, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            nn.init.normal_(h[2].weight, std=0.01)
            nn.init.zeros_(h[2].bias)
            heads.append(h)
        self.heads = nn.ModuleList(heads)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        p = self.proj(f)
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            pp = F.pad(p, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(pp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-pp, (dy, dx), stride=1)
            outs.append(head(torch.cat([p, mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
