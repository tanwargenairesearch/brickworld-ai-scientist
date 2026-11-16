import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 2e-3, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 32, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.25}

COLH = 96
COL = 48
CTX = 32
PROJ = 32
HID = 24
GROUPS = 4


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class WinHead(nn.Module):
    def __init__(self, c, n, proj, hid):
        super().__init__()
        self.proj = nn.Conv2d(c, proj, 1)
        self.win = nn.ModuleList([nn.Conv2d(proj, hid, 3, padding=1) for _ in range(n)])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in range(n)])
        self.inter = nn.Conv2d(c, n, 1)
        self.anchor = nn.Conv2d(c, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(n))

    def forward(self, f):
        p = F.relu(self.proj(f))
        res = []
        for win, out in zip(self.win, self.outs):
            res.append(out(F.relu(win(p))))
        s = torch.cat(res, dim=1) + self.inter(f) + self.anchor(f)
        s = s + self.tbias.view(1, -1, 1, 1)
        return s.flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        n = len(spec.footprints)
        self.col = nn.Sequential(
            nn.Conv2d(cin, COLH, 1), nn.GroupNorm(GROUPS, COLH), nn.ReLU(),
            nn.Conv2d(COLH, COLH, 1), nn.ReLU(),
            nn.Conv2d(COLH, COL, 1), nn.ReLU())
        self.ctx = nn.Conv2d(COL, CTX, 3, padding=1)
        self.norm = nn.GroupNorm(GROUPS, CTX)
        c = COL + CTX
        self.head = WinHead(c, n, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        for m in self.head.outs:
            init_layer(m, 0.01)
        init_layer(self.head.inter, 0.01)
        init_layer(self.head.anchor, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c = self.col(obs)
        x = F.relu(self.norm(self.ctx(c)))
        h = torch.cat([c, x], dim=1)
        place = self.head(h)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
