"""Two-layer column MLP stem + 3x3 conv body, stem skip into footprint-pooled placement head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 4, "lr": 0.001}

STEM = 64
WIDTH = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem1 = nn.Conv2d(spec.in_channels, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, STEM, 1)
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH + STEM, WIDTH, 1)
        self.type_heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = F.relu(self.stem1(obs))
        s = F.relu(self.stem2(s))
        f = F.relu(self.c1(s))
        f = F.relu(self.c2(f))
        h = F.relu(self.proj(torch.cat([f, s], dim=1)))
        outs = []
        for (dy, dx), head in zip(self.fps, self.type_heads):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            outs.append(head(torch.cat([mx, mn, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
