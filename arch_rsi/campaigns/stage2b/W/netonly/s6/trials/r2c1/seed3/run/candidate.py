import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 36
BLOCKS = 3
K = 16
HID = 32


def fp_pool(u, dy, dx):
    if dy == 1 and dx == 1:
        return u, u
    p = F.pad(u, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    return mx, mn


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.mid_glob = nn.Linear(2 * WIDTH, WIDTH)
        self.cell = nn.Conv2d(WIDTH, K, 1)
        self.type_hid = nn.ModuleList([nn.Conv2d(2 * K, HID, 1) for _ in self.fps])
        self.type_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        x = self.stem(obs)
        for i, conv in enumerate(self.blocks):
            x = x + conv(F.relu(x))
            if i == 0:
                r = F.relu(x)
                g0 = self.mid_glob(torch.cat([r.mean((2, 3)), r.amax((2, 3))], dim=1))
                x = x + g0[:, :, None, None]
        f = F.relu(x)
        u = self.cell(f)
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            mx, mn = fp_pool(u, dy, dx)
            h = F.relu(self.type_hid[t](torch.cat([mx, mn], dim=1)))
            outs.append(self.type_out[t](h).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
