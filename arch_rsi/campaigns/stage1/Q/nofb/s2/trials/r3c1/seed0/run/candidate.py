import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
BLOCKS = 2
SKIP = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(BLOCKS)])
        self.globs = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(BLOCKS)])
        self.obs_proj = nn.Conv2d(spec.in_channels, SKIP, 1)
        hc = WIDTH + SKIP
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.fp_conv = nn.ModuleList([nn.Conv2d(hc, HID, (dy, dx)) for dy, dx in self.fps])
        self.fp_max = nn.ModuleList([nn.Conv2d(hc, HID, 1) for _ in self.fps])
        self.fp_out = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        for conv, gl in zip(self.convs, self.globs):
            g = gl(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
            h = h + F.relu(conv(F.relu(h + g[:, :, None, None])))
        z = torch.cat([h, F.relu(self.obs_proj(obs))], dim=1)
        outs = []
        for (dy, dx), conv, mx, out in zip(self.fps, self.fp_conv, self.fp_max, self.fp_out):
            zp = F.pad(z, (0, dx - 1, 0, dy - 1))
            pooled = F.max_pool2d(zp, (dy, dx), stride=1)
            u = F.relu(conv(zp) + mx(pooled))
            outs.append(out(u))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
