import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'n_envs': 128, 'rollout_len': 16, 'n_minibatches': 8}

STEM = 64
COL = 48
CTX = 32
SKIP = 32
PROJ = 32
H1 = 48
H2 = 48
H3 = 32
G = 64


def orth(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def coord_channels(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([ys, xs], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(nn.Conv2d(cin + 2, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.ctx = nn.Conv2d(COL, CTX, 3, padding=1)
        self.gn = nn.GroupNorm(4, CTX)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.proj = nn.Conv2d(COL + CTX + SKIP, PROJ, 1)
        self.m1 = nn.ModuleList([nn.Conv2d(3 * PROJ, H1, 1) for _ in self.fps])
        self.m2 = nn.ModuleList([nn.Conv2d(H1, H2, 1) for _ in self.fps])
        self.m3 = nn.ModuleList([nn.Conv2d(H2, H3, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H3, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * COL + 2 * CTX + SKIP, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for m in self.outs:
            orth(m, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        s = self.stem(torch.cat([obs, coord_channels(obs)], dim=1))
        h = F.relu(self.gn(self.ctx(s)))
        sk = F.relu(self.skip(obs))
        p = self.proj(torch.cat([s, h, sk], dim=1))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            z = F.relu(self.m1[i](torch.cat([mx, mn, av], dim=1)))
            z = F.relu(self.m2[i](z))
            z = F.relu(self.m3[i](z))
            res.append(self.outs[i](z))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([s.mean((2, 3)), s.amax((2, 3)), h.mean((2, 3)), h.amax((2, 3)), sk.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
