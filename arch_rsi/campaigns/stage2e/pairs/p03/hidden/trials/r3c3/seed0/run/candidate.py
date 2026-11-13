import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.98}

WIDTH = 48
BLOCKS = 3
PW = 24
HID = 24


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv2d(w, w, 3, padding=1)
        self.ctx = nn.Linear(2 * w, w)

    def forward(self, h):
        g = self.ctx(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        return h + F.relu(self.conv(h) + g[:, :, None, None])


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(WIDTH) for _ in range(BLOCKS)])
        self.proj = nn.Sequential(nn.Conv2d(WIDTH, PW, 1), nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PW + WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = self.blocks(self.stem(obs))
        q = self.proj(f)
        outs = []
        for (dy, dx), head in zip(self.fps, self.heads):
            qp = F.pad(q, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(qp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-qp, (dy, dx), stride=1)
            av = F.avg_pool2d(qp, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av, f], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
