import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
GW = 64
PROJ = 32
HID = 16


def fp_pool(h, dy, dx):
    if dy == 1 and dx == 1:
        return h, h
    p = F.pad(h, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(p, (dy, dx), stride=1)
    mn = -F.max_pool2d(-p, (dy, dx), stride=1)
    return mx, mn


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GW), nn.ReLU())
        self.ctx = nn.Linear(GW, WIDTH)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.skip = nn.Conv2d(spec.in_channels, PROJ, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * PROJ, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.done_head = nn.Linear(GW, 1)
        self.value_head = nn.Linear(GW, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(f + self.ctx(g)[:, :, None, None])
        q = F.relu(self.proj(h) + self.skip(obs))
        outs = []
        for fp, head in zip(self.fps, self.heads):
            mx, mn = fp_pool(q, fp[0], fp[1])
            outs.append(head(torch.cat([mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
