"""Column MLP stem + 3x3 trunk; placement logits = generic footprint-pooled head
+ sum over layers of (learned soft landing-layer gate) x (footprint-pooled per-layer score)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 4, "lr": 0.001}

STEM = 48
WIDTH = 48
K = 8
PROJ = 24


def fp_pool(h, dy, dx):
    hp = F.pad(h, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(hp, (dy, dx), stride=1)
    mn = -F.max_pool2d(-hp, (dy, dx), stride=1)
    av = F.avg_pool2d(hp, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        self.stem1 = nn.Conv2d(spec.in_channels, 64, 1)
        self.stem2 = nn.Conv2d(64, STEM, 1)
        self.c1 = nn.Conv2d(STEM, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.gate = nn.Conv2d(STEM, self.Z, 1)
        self.layer_feat = nn.Conv2d(WIDTH + STEM, self.Z * K, 1)
        self.proj = nn.Conv2d(WIDTH, PROJ, 1)
        self.type_heads = nn.ModuleList([nn.Conv2d(3 * PROJ, 1, 1) for _ in self.fps])
        self.layer_heads = nn.ModuleList([nn.Linear(3 * K, 1) for _ in self.fps])
        self.layer_bias = nn.Parameter(torch.zeros(len(self.fps), self.Z))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        s = F.relu(self.stem1(obs))
        s = F.relu(self.stem2(s))
        f = F.relu(self.c1(s))
        f = F.relu(self.c2(f))
        a = torch.sigmoid(self.gate(s))
        u = F.relu(self.layer_feat(torch.cat([f, s], dim=1)))
        h = F.relu(self.proj(f))
        outs = []
        for t, (dy, dx) in enumerate(self.fps):
            mx, mn, av = fp_pool(h, dy, dx)
            generic = self.type_heads[t](torch.cat([mx, mn, av], dim=1)).squeeze(1)
            umx, umn, uav = fp_pool(u, dy, dx)
            pu = torch.stack([umx, umn, uav], dim=1).view(B, 3, Z, K, Y, X)
            pu = pu.permute(0, 2, 4, 5, 1, 3).reshape(B, Z, Y, X, 3 * K)
            score = self.layer_heads[t](pu).squeeze(-1) + self.layer_bias[t].view(1, Z, 1, 1)
            ap = F.pad(a, (0, dx - 1, 0, dy - 1))
            A = F.max_pool2d(ap, (dy, dx), stride=1)
            A_prev = torch.cat([torch.ones_like(A[:, :1]), A[:, :-1]], dim=1)
            p = A_prev * (1.0 - A)
            outs.append(generic + (p * score).sum(1))
        place = torch.stack(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
