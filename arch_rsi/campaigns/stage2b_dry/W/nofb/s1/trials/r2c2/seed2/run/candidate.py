"""Conv body with a global-context residual block and per-type footprint max/min pooling head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HID = 16
RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.c1 = nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.footprints
        ])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = F.relu(self.c1(obs))
        f = F.relu(self.c2(f))
        c = self.ctx(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        f = f + F.relu(self.c3(f) + c[:, :, None, None])
        outs = []
        for (dy, dx), head in zip(self.footprints, self.heads):
            if dy > 1 or dx > 1:
                p = F.pad(f, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(p, (dy, dx), stride=1)
                mn = -F.max_pool2d(-p, (dy, dx), stride=1)
            else:
                mx = f
                mn = f
            outs.append(head(torch.cat([mx, mn], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
