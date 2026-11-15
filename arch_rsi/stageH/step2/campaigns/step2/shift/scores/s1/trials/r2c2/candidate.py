import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

H1 = 96
H2 = 64
PROJ = 24
CTX = 24
MIXH = 48
MIXH2 = 32


def pool3(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


def maxfp(c, dy, dx):
    if dy * dx == 1:
        return c
    q = F.pad(c, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(q, (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.col = nn.Sequential(
            nn.Conv2d(C, H1, 1), nn.ReLU(),
            nn.Conv2d(H1, H2, 1), nn.ReLU())
        self.proj = nn.Conv2d(H2, PROJ, 1)
        self.ctx1 = nn.Conv2d(H2, CTX, 3, padding=1)
        self.ctx2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.mix1 = nn.ModuleList([nn.Conv2d(3 * PROJ + 2 * CTX, MIXH, 1) for _ in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(MIXH, MIXH2, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MIXH2, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(
            nn.Linear(2 * (H2 + CTX), 96), nn.ReLU(),
            nn.Linear(96, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        h = self.col(obs)
        p = self.proj(h)
        c = F.relu(self.ctx1(h))
        c = F.relu(c + self.ctx2(c))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            feats = pool3(p, dy, dx) + [c, maxfp(c, dy, dx)]
            z = F.relu(self.mix1[i](torch.cat(feats, dim=1)))
            z = F.relu(self.mix2[i](z))
            res.append(self.outs[i](z))
        place = torch.cat(res, dim=1).flatten(1)
        hc = torch.cat([h, c], dim=1)
        g = self.glob(torch.cat([hc.mean((2, 3)), hc.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
