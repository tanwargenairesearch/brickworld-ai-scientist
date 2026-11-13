import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HW = 24


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for (a, b) in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(WIDTH, HW, (dy, dx)), nn.ReLU(), nn.Conv2d(HW, 1, 1))
            for (dy, dx) in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = h + F.relu(self.c2(h))
        c = self.ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        h = h + F.relu(self.c3(h + c[:, :, None, None]))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            q = F.pad(h, (0, dx - 1, 0, dy - 1))
            outs.append(head(q))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
