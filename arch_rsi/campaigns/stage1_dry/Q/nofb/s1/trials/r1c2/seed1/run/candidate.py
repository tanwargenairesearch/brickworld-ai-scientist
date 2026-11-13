import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 44


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c_in = spec.in_channels
        self.stem = nn.Conv2d(c_in, WIDTH, 3, padding=1)
        self.conv1 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Linear(2 * WIDTH, WIDTH)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = F.relu(self.stem(obs))
        h = h + F.relu(self.conv1(h))
        g = F.relu(self.ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)))
        h = h + g[:, :, None, None]
        f = h + F.relu(self.conv2(h))
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            p = F.pad(f, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(p, (dy, dx), stride=1)
            mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            av = F.avg_pool2d(p, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av], dim=1)).flatten(1))
        place = torch.cat(outs, dim=1)
        z = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(z)], dim=1)
        return logits, self.value_head(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
