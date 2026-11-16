import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 1e-3, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 32, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2}

COLH = 96
COL = 48
W = 40
PROJ = 32
HID = 12
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


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(G, c)
        self.c1 = nn.Conv2d(c, c, 3)
        self.n2 = nn.GroupNorm(G, c)
        self.c2 = nn.Conv2d(c, c, 3)

    def forward(self, x):
        h = self.c1(F.pad(F.relu(self.n1(x)), (1, 1, 1, 1), mode='circular'))
        h = self.c2(F.pad(F.relu(self.n2(h)), (1, 1, 1, 1), mode='circular'))
        return x + h


class WinHead(nn.Module):
    def __init__(self, c, n, proj, hid):
        super().__init__()
        self.proj = nn.Conv2d(c, proj, 1)
        self.win = nn.Conv2d(proj, n * hid, 3)
        self.out = nn.Conv2d(n * hid, n, 1, groups=n)

    def forward(self, f):
        p = F.relu(self.proj(f))
        p = F.pad(p, (1, 1, 1, 1), mode='circular')
        return self.out(F.relu(self.win(p)))


class DenseHead(nn.Module):
    def __init__(self, c, y, x, n_out, ch, hid):
        super().__init__()
        self.proj = nn.Conv2d(c, ch, 1)
        self.l1 = nn.Linear(ch * y * x, hid)
        self.l2 = nn.Linear(hid, n_out)

    def forward(self, f):
        return self.l2(F.relu(self.l1(F.relu(self.proj(f)).flatten(1))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        n = len(spec.footprints)
        self.col = nn.Sequential(
            nn.Conv2d(cin, COLH, 1), nn.GroupNorm(G, COLH), nn.ReLU(),
            nn.Conv2d(COLH, COL, 1), nn.ReLU())
        self.inp = nn.Conv2d(COL, W, 1)
        self.res = Res(W)
        self.norm = nn.GroupNorm(G, W)
        c = COL + W
        self.head = WinHead(c, n, PROJ, HID)
        self.inter = nn.Conv2d(c, n, 1)
        self.anchor = nn.Conv2d(c, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(n))
        self.dense = DenseHead(c, spec.Y, spec.X, spec.n_actions, 8, 256)
        self.glob = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        init_layer(self.head.out, 0.01)
        init_layer(self.inter, 0.01)
        init_layer(self.anchor, 0.01)
        init_layer(self.dense.l2, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c0 = self.col(add_coords(obs))
        f = F.relu(self.norm(self.res(self.inp(c0))))
        h = torch.cat([c0, f], dim=1)
        s = self.head(h) + self.inter(h) + self.anchor(h) + self.tbias.view(1, -1, 1, 1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([s.flatten(1), self.done_head(g)], dim=1) + self.dense(h)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
