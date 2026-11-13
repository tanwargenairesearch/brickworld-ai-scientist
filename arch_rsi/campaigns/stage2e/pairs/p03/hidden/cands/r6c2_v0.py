import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95}

WIDTH = 64
PW = 32
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        c = spec.in_channels
        self.body = nn.Sequential(
            nn.Conv2d(c, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv2d(WIDTH, PW, 1), nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * PW + WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
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
