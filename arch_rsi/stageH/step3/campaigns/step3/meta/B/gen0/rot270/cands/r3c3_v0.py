import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.9}

CELL = 64
CTX = 32
PROJ = 32
HID = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.cell1 = nn.Conv2d(spec.in_channels, CELL, 1)
        self.cell2 = nn.Conv2d(CELL, CELL, 1)
        self.ctx = nn.Conv2d(CELL, CTX, 3, padding=1)
        self.proj = nn.Conv2d(CELL + CTX, PROJ, 1)
        self.mix1 = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.mix2 = nn.ModuleList([nn.Conv2d(HID, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * (CELL + CTX), G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        c = F.relu(self.cell1(obs))
        c = F.relu(self.cell2(c))
        k = F.relu(self.ctx(c))
        feat = torch.cat([c, k], dim=1)
        p = self.proj(feat)
        res = []
        for (dy, dx), m1, m2, out in zip(self.fps, self.mix1, self.mix2, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(m1(torch.cat([mx, mn, av], dim=1)))
            h = F.relu(m2(h))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([feat.mean((2, 3)), feat.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
