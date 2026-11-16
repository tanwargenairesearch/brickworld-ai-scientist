import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 3, "lr": 0.0007, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

COL = 64
K = 24
W = 32
H = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            nn.Conv2d(spec.in_channels, COL, 1), nn.ReLU(),
            nn.Conv2d(COL, COL, 1), nn.ReLU())
        self.key = nn.Conv2d(COL, K, 1)
        self.c1 = nn.Conv2d(COL, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * K + W, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W + 2 * K, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = self.col(obs)
        e = self.key(h)
        c = F.relu(self.c1(h))
        c = F.relu(c + self.c2(c))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn = e, e
            else:
                q = F.pad(e, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            s = out(F.relu(mix(torch.cat([mx, mn, c], dim=1))))
            res.append(s)
        place = torch.cat(res, dim=1).flatten(1)
        g = torch.cat([c.mean((2, 3)), c.amax((2, 3)), e.mean((2, 3)), e.amax((2, 3))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
