import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.res1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.res2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.glob1 = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.ctx = nn.Linear(WIDTH, WIDTH)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.glob2 = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        f = f + F.relu(self.res1(f))
        g1 = self.glob1(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = f + self.ctx(g1)[:, :, None, None]
        f = f + F.relu(self.res2(f))
        outs = []
        for fp, head in zip(self.fps, self.heads):
            dy, dx = fp
            if dy == 1 and dx == 1:
                mx = f
                mn = f
            else:
                p = F.pad(f, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, (dy, dx), stride=1)
                mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob2(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
