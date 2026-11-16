import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 1e-3, 'n_envs': 256, 'rollout_len': 8, 'n_minibatches': 8, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2}

COLH = 160
COL = 80
W = 48
PROJ = 48
FHID = 32
WPROJ = 32
WHID = 16
G = 4


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b = obs.shape[0]
    y = obs.shape[2]
    x = obs.shape[3]
    yy = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xx = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, yy, xx], dim=1)


class FootHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.foot = nn.ModuleList([nn.Conv2d(proj, hid, (dy, dx)) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f, gb):
        p = F.relu(self.proj(f) + gb[:, :, None, None])
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
        cin = spec.in_channels + 2
        n = len(spec.footprints)
        self.col = nn.Sequential(
            nn.Conv2d(cin, COLH, 1), nn.GroupNorm(G, COLH), nn.ReLU(),
            nn.Conv2d(COLH, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, W, 3, padding=1)
        self.n1 = nn.GroupNorm(G, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(G, W)
        c = COL + W
        self.glob = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU())
        self.gproj = nn.Linear(64, PROJ)
        self.fhead = FootHead(c, spec.footprints, PROJ, FHID)
        self.whead = WinHead(c, n, WPROJ, WHID)
        self.inter = nn.Conv2d(c, n, 1)
        self.anchor = nn.Conv2d(c, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(n))
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        for m in self.fhead.outs:
            init_layer(m, 0.01)
        init_layer(self.whead.out, 0.01)
        init_layer(self.inter, 0.01)
        init_layer(self.anchor, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c0 = self.col(add_coords(obs))
        x = F.relu(self.n1(self.c1(c0)))
        f = F.relu(self.n2(x + self.c2(x)))
        h = torch.cat([c0, f], dim=1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        s = self.fhead(h, self.gproj(g)) + self.whead(h) + self.inter(h) + self.anchor(h) + self.tbias.view(1, -1, 1, 1)
        logits = torch.cat([s.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
