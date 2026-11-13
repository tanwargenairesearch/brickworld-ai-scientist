"""r2c2 with a second 1x1 hidden layer in each footprint head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 48
HEAD = 32
RECIPE = {"n_envs": 128, "ppo_epochs": 2, "n_minibatches": 16}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.body = nn.Sequential(
            nn.Conv2d(c, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.foot = [(int(a), int(b)) for a, b in spec.footprints]
        self.heads = nn.ModuleList()
        self.gproj = nn.ModuleList()
        self.mids = nn.ModuleList()
        self.outs = nn.ModuleList()
        for dy, dx in self.foot:
            self.heads.append(nn.Conv2d(WIDTH + c, HEAD, (dy, dx)))
            self.gproj.append(nn.Linear(WIDTH, HEAD))
            self.mids.append(nn.Conv2d(HEAD, HEAD, 1))
            self.outs.append(nn.Conv2d(HEAD, 1, 1))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = torch.cat([f, obs], dim=1)
        parts = []
        for (dy, dx), conv, gp, mid, out in zip(self.foot, self.heads, self.gproj, self.mids, self.outs):
            p = F.pad(h, (0, dx - 1, 0, dy - 1))
            z = F.relu(conv(p) + gp(g)[:, :, None, None])
            z = F.relu(mid(z))
            parts.append(out(z).flatten(1))
        logits = torch.cat(parts + [self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
