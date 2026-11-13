import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HEAD = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.conv1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.fp_hid = nn.ModuleList([nn.Conv2d(WIDTH, HEAD, (dy, dx)) for dy, dx in self.fps])
        self.fp_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.conv2(F.relu(self.conv1(obs))))
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        c = self.ctx(pooled)
        fc = F.relu(f + c.unsqueeze(2).unsqueeze(3))
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            h = F.relu(self.fp_hid[i](F.pad(fc, (0, dx - 1, 0, dy - 1))))
            outs.append(self.fp_out[i](h))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
