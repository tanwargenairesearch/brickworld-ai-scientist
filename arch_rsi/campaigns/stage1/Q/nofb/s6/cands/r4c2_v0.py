import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HID = 32
BLOCKS = 2
GW = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.bcast = nn.Linear(GW, WIDTH)
        self.cell = nn.Conv2d(WIDTH, HID, 1)
        self.type_heads = nn.ModuleList(
            [nn.Conv2d(WIDTH + 2 * HID, 1, 1) for _ in self.footprints])
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)
        for h in list(self.type_heads) + [self.done_head]:
            nn.init.orthogonal_(h.weight, gain=0.01)
            nn.init.zeros_(h.bias)

    def forward(self, obs, mask):
        f = torch.relu(self.stem(obs))
        for conv in self.blocks:
            f = f + torch.relu(conv(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = torch.relu(f + self.bcast(g).unsqueeze(-1).unsqueeze(-1))
        c = self.cell(f)
        outs = []
        for t, (dy, dx) in enumerate(self.footprints):
            cp = F.pad(c, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(cp, (dy, dx), stride=1)
            av = F.avg_pool2d(cp, (dy, dx), stride=1)
            outs.append(self.type_heads[t](torch.cat([f, mx, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
