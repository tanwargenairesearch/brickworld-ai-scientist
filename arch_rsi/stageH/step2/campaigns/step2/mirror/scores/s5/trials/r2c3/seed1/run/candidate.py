import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "gamma": 0.97, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
MIX = 24


def pool_fp(q, dy, dx):
    if dy * dx == 1:
        return q, q, q
    r = F.pad(q, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(r, (dy, dx), stride=1)
    mn = -F.max_pool2d(-r, (dy, dx), stride=1)
    av = F.avg_pool2d(r, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.proj = nn.Conv2d(WIDTH, self.Z * PROJ, 1)
        self.own = nn.Conv2d(5, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, MIX, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(MIX, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def own_input(self, obs):
        B, C, Y, X = obs.shape
        Z = self.Z
        build = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        below = F.pad(build[:, :Z - 1], (0, 0, 0, 0, 1, 0), value=1.0)
        own = torch.stack([build, tgt, miss, extra, below], dim=2)
        return own.reshape(B * Z, 5, Y, X)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        p = self.proj(f).reshape(B, Z, PROJ, Y, X).reshape(B * Z, PROJ, Y, X)
        p = p + self.own(self.own_input(obs))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = pool_fp(p, dy, dx)
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        sc = torch.cat(res, dim=1).reshape(B, Z, len(self.fps), Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
