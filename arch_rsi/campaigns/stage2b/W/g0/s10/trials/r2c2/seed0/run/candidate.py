"""r1c2 footprint-pooled placement head trained with 2 PPO epochs for about 1.9x the samples."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HID = 32

RECIPE = {"ppo_epochs": 2, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.type_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * WIDTH, HID, 1), nn.ReLU(), nn.Conv2d(HID, 1, 1))
            for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def pool(self, f, dy, dx):
        if dy == 1 and dx == 1:
            return f, f
        p = F.pad(f, (0, dx - 1, 0, dy - 1))
        return F.max_pool2d(p, (dy, dx), stride=1), F.avg_pool2d(p, (dy, dx), stride=1)

    def forward(self, obs, mask):
        f = self.body(obs)
        outs = []
        for (dy, dx), head in zip(self.fps, self.type_heads):
            pmax, pmean = self.pool(f, dy, dx)
            outs.append(head(torch.cat([f, pmax, pmean], dim=1)).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
