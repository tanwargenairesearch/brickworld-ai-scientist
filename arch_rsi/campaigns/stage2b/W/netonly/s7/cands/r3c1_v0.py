import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 36
N_RES = 3
EMB = 16
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.embed = nn.Conv2d(WIDTH, EMB, 1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * EMB, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in range(spec.n_types)
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        c = self.ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h = h + c[:, :, None, None]
        for conv in self.res:
            h = h + F.relu(conv(h))
        e = self.embed(h)
        outs = []
        for t in range(len(self.fps)):
            dy, dx = self.fps[t]
            ep = F.pad(e, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(ep, kernel_size=(dy, dx), stride=1)
            av = F.avg_pool2d(ep, kernel_size=(dy, dx), stride=1)
            outs.append(self.type_heads[t](torch.cat([mx, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
