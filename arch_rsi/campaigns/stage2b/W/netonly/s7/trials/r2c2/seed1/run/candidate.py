import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 56
DEPTH = 2
EMB = 16
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        layers, c_in = [], spec.in_channels
        for _ in range(DEPTH):
            layers += [nn.Conv2d(c_in, WIDTH, 3, padding=1), nn.ReLU()]
            c_in = WIDTH
        self.body = nn.Sequential(*layers)
        self.embed = nn.Conv2d(WIDTH, EMB, 1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * EMB, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in range(spec.n_types)
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        e = self.embed(f)
        outs = []
        for t in range(len(self.fps)):
            dy, dx = self.fps[t]
            ep = F.pad(e, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(ep, kernel_size=(dy, dx), stride=1)
            mn = -F.max_pool2d(-ep, kernel_size=(dy, dx), stride=1)
            av = F.avg_pool2d(ep, kernel_size=(dy, dx), stride=1)
            outs.append(self.type_heads[t](torch.cat([mx, mn, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
