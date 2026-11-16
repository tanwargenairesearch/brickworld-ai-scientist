import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 2e-3, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 64, 'ppo_epochs': 3, 'ent_coef': 0.01, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_eps': 0.2}

COLH = 96
COL = 48
W = 40
D = 64
PROJ = 32
HID = 16
G = 4


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        n = len(spec.footprints)
        self.col = nn.Sequential(
            nn.Conv2d(cin, COLH, 1), nn.GroupNorm(G, COLH), nn.ReLU(),
            nn.Conv2d(COLH, COL, 1), nn.ReLU())
        self.e1 = nn.Conv2d(COL, W, 3, padding=1)
        self.n1 = nn.GroupNorm(G, W)
        self.d1 = nn.Conv2d(W, D, 3, stride=2, padding=1)
        self.nd1 = nn.GroupNorm(G, D)
        self.d2 = nn.Conv2d(D, D, 3, stride=2, padding=1)
        self.nd2 = nn.GroupNorm(G, D)
        self.gl = nn.Linear(D, D)
        self.u1 = nn.Conv2d(D, D, 1)
        self.u0 = nn.Conv2d(D, W, 1)
        self.nu = nn.GroupNorm(G, W)
        c = COL + W
        self.hproj = nn.Conv2d(c, PROJ, 1)
        self.wins = nn.ModuleList([nn.Conv2d(PROJ, HID, 3, padding=1) for _ in range(n)])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in range(n)])
        self.inter = nn.Conv2d(c, n, 1)
        self.tbias = nn.Parameter(torch.zeros(n))
        gdim = 2 * c + D
        self.glob = nn.Sequential(nn.Linear(gdim, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(gdim, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        for m in self.outs:
            init_layer(m, 0.01)
        init_layer(self.inter, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c0 = self.col(obs)
        e = F.relu(self.n1(self.e1(c0)))
        a = F.relu(self.nd1(self.d1(e)))
        b = F.relu(self.nd2(self.d2(a)))
        g = F.relu(self.gl(b.mean((2, 3))))
        b_up = F.interpolate(b, size=(a.shape[2], a.shape[3]), mode='nearest')
        a2 = F.relu(self.u1(a + b_up) + g[:, :, None, None])
        a_up = F.interpolate(a2, size=(e.shape[2], e.shape[3]), mode='nearest')
        f = F.relu(e + self.nu(self.u0(a_up)))
        h = torch.cat([c0, f], dim=1)
        p = F.relu(self.hproj(h))
        res = []
        for win, out in zip(self.wins, self.outs):
            res.append(out(F.relu(win(p))))
        s = torch.cat(res, dim=1) + self.inter(h) + self.tbias.view(1, -1, 1, 1)
        place = s.flatten(1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3)), g], dim=1)
        gg = self.glob(pooled)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
