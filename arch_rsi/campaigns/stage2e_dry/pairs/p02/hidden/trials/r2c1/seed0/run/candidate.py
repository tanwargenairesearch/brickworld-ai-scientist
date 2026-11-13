"""Column stem + 3x3 conv body + per-brick-type learned footprint-shaped conv placement head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 4, "lr": 0.001}

STEM = 64
WIDTH = 48
HEAD = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem = nn.Conv2d(spec.in_channels, STEM, 1)
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.fp_convs = nn.ModuleList([nn.Conv2d(WIDTH, HEAD, (dy, dx)) for (dy, dx) in self.fps])
        self.fp_out = nn.ModuleList([nn.Conv2d(HEAD, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        f = F.relu(self.stem(obs))
        f = F.relu(self.c1(f))
        f = F.relu(self.c2(f))
        outs = []
        for (dy, dx), conv, out in zip(self.fps, self.fp_convs, self.fp_out):
            fp = F.pad(f, (0, dx - 1, 0, dy - 1))
            outs.append(out(F.relu(conv(fp))))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
