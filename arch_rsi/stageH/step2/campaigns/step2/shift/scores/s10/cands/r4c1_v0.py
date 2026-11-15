import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 8, "n_minibatches": 8, "ppo_epochs": 3, "ent_coef": 0.005}

USE_MASK = False
STEM = 64
CELL = 48
LH = 32
LF = 16
VH = 48
VO = 32
G = 64


def shift(t, i, j):
    if i == 0 and j == 0:
        return t
    Y = t.shape[2]
    X = t.shape[3]
    q = F.pad(t, (0, j, 0, i))
    return q[:, :, i:i + Y, j:j + X]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + (self.T if USE_MASK else 0)
        self.stem = nn.Sequential(nn.Conv2d(cin, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, CELL, 1), nn.ReLU())
        self.ctx = nn.Conv2d(CELL, CELL, 3, padding=1)
        self.land = nn.Sequential(nn.Conv2d(cin, LH, 1), nn.ReLU(), nn.Conv2d(LH, LF, 1), nn.ReLU())
        self.v1 = nn.Conv2d(CELL + LF, VH, 1)
        self.v2 = nn.Conv2d(VH, VO, 1)
        self.outs = nn.ModuleList([nn.Conv2d(3 * VO, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * CELL, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        inp = obs
        if USE_MASK:
            m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
            inp = torch.cat([obs, m], dim=1)
        s = self.stem(inp)
        c = F.relu(s + self.ctx(s))
        lf = self.land(inp)
        xs = []
        for (dy, dx) in self.fps:
            if dy * dx == 1:
                l = lf
            else:
                l = F.max_pool2d(F.pad(lf, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)
            for i in range(dy):
                for j in range(dx):
                    xs.append(torch.cat([shift(c, i, j), l], dim=1))
        n = len(xs)
        h = torch.cat(xs, dim=0)
        h = F.relu(self.v2(F.relu(self.v1(h))))
        h = h.view(n, B, VO, Y, X)
        res = []
        k = 0
        for (dy, dx), out in zip(self.fps, self.outs):
            cnt = dy * dx
            hh = h[k:k + cnt]
            k += cnt
            agg = torch.cat([hh.amax(0), hh.amin(0), hh.mean(0)], dim=1)
            res.append(out(agg))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([c.mean((2, 3)), c.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
