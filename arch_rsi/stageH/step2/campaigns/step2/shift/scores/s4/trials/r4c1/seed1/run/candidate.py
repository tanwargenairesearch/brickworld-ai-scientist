import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "lr": 0.001, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.95}

S1 = 64
S2 = 48
W = 32
P = 40
H = 40
GP = 32
GH = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = int(spec.in_channels)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(C, S1, 1), nn.ReLU(), nn.Conv2d(S1, S2, 1), nn.ReLU())
        self.c1 = nn.Conv2d(S2, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.cell = nn.Conv2d(S2 + W, P, 1)
        self.gp = nn.Conv2d(S2 + W, GP, 1)
        self.glob = nn.Sequential(nn.Linear(2 * GP, GH), nn.ReLU(), nn.Linear(GH, GH), nn.ReLU())
        self.h1 = nn.ModuleList([nn.Conv2d(dy * dx * P, H, 1) for dy, dx in self.fps])
        self.gb = nn.ModuleList([nn.Linear(GH, H) for _ in self.fps])
        self.h2 = nn.ModuleList([nn.Conv2d(H, H, 1) for _ in self.fps])
        self.h3 = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        for l in self.h3:
            nn.init.orthogonal_(l.weight, gain=0.01)
            nn.init.zeros_(l.bias)
        self.done_head = nn.Linear(GH, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        sf = torch.cat([s, f], dim=1)
        g = F.relu(self.cell(sf))
        q = self.gp(sf)
        gl = self.glob(torch.cat([q.amax((2, 3)), q.mean((2, 3))], dim=1))
        res = []
        for (dy, dx), l1, lb, l2, l3 in zip(self.fps, self.h1, self.gb, self.h2, self.h3):
            qq = F.pad(g, (0, dx - 1, 0, dy - 1))
            parts = [qq[:, :, oy:oy + Y, ox:ox + X] for oy in range(dy) for ox in range(dx)]
            h = torch.cat(parts, dim=1)
            h = F.relu(l1(h) + lb(gl)[:, :, None, None])
            h = F.relu(h + l2(h))
            res.append(l3(h).flatten(1))
        place = torch.cat(res, dim=1)
        logits = torch.cat([place, self.done_head(gl)], dim=1)
        return logits, self.value_head(gl).squeeze(1)


def build_policy(spec):
    return Net(spec)
