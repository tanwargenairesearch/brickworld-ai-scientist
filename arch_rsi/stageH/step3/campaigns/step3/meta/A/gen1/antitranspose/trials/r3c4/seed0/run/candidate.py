import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 2e-3, 'n_envs': 1024, 'rollout_len': 16, 'n_minibatches': 64, 'ppo_epochs': 2, 'ent_coef': 0.05, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.3}

COLH = 96
COL = 32
W = 24
ITERS = 6
PROJ = 24
FHID = 16
WHID = 8
G = 4


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.foot = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = F.relu(self.proj(f))
        res = []
        for (dy, dx), foot, out in zip(self.fps, self.foot, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(foot(q))))
        return torch.cat(res, dim=1)


class WinHead(nn.Module):
    def __init__(self, c, n, proj, hid):
        super().__init__()
        self.proj = nn.Conv2d(c, proj, 1)
        self.win = nn.Conv2d(proj, n * hid, 3, padding=1)
        self.out = nn.Conv2d(n * hid, n, 1, groups=n)

    def forward(self, f):
        p = F.relu(self.proj(f))
        return self.out(F.relu(self.win(p)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        n = len(spec.footprints)
        self.col = nn.Sequential(
            nn.Conv2d(cin, COLH, 1), nn.GroupNorm(G, COLH), nn.ReLU(),
            nn.Conv2d(COLH, COL, 1), nn.ReLU())
        self.inj = nn.Conv2d(COL, W, 1)
        self.rn = nn.GroupNorm(G, W)
        self.rc = nn.Conv2d(W, W, 3, padding=1)
        self.norm = nn.GroupNorm(G, W)
        c = COL + W
        self.fhead = FootHead(c, spec.footprints, PROJ, FHID)
        self.whead = WinHead(c, n, PROJ, WHID)
        self.inter = nn.Conv2d(c, n, 1)
        self.anchor = nn.Conv2d(c, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(n))
        self.glob = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        init_layer(self.rc, 1.0)
        for m in self.fhead.outs:
            init_layer(m, 0.01)
        init_layer(self.whead.out, 0.01)
        init_layer(self.inter, 0.01)
        init_layer(self.anchor, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c0 = self.col(obs)
        e = self.inj(c0)
        x = e
        for _ in range(ITERS):
            x = e + self.rc(F.relu(self.rn(x)))
        f = F.relu(self.norm(x))
        h = torch.cat([c0, f], dim=1)
        s = self.fhead(h) + self.whead(h) + self.inter(h) + self.anchor(h) + self.tbias.view(1, -1, 1, 1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([s.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
