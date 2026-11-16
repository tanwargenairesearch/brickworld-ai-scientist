import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 1e-3, 'n_envs': 64, 'rollout_len': 64, 'n_minibatches': 64, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.995, 'gae_lambda': 0.95, 'clip_eps': 0.2}

COLH = 64
COL = 32
W = 32
MPROJ = 16
MHID = 48
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


class FilmHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        n = len(footprints)
        span = max(max(int(a), int(b)) for a, b in footprints)
        self.k = span + 2
        self.proj = nn.Conv2d(c, proj, 1)
        self.win = nn.Conv2d(proj, hid, self.k)
        self.scale = nn.Parameter(torch.randn(n, hid) * 0.1)
        self.shift = nn.Parameter(torch.randn(n, hid) * 0.1)
        self.wout = nn.Parameter(torch.randn(n, hid) * (0.01 / math.sqrt(hid)))
        self.bout = nn.Parameter(torch.zeros(n))

    def forward(self, f):
        p = F.relu(self.proj(f))
        k = self.k
        q = F.pad(p, (1, k - 2, 1, k - 2))
        a = self.win(q).unsqueeze(1)
        h = F.relu(a * (1.0 + self.scale[None, :, :, None, None]) + self.shift[None, :, :, None, None])
        return (h * self.wout[None, :, :, None, None]).sum(2) + self.bout[None, :, None, None]


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
        c = COL + W
        self.film = FilmHead(c, spec.footprints, MPROJ, MHID)
        self.inter = nn.Conv2d(c, n, 1)
        self.anchor = nn.Conv2d(c, 1, 1)
        self.tbias = nn.Parameter(torch.zeros(n))
        self.glob = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_layer(m, math.sqrt(2))
        init_layer(self.inter, 0.01)
        init_layer(self.anchor, 0.01)
        init_layer(self.done_head, 0.01)
        init_layer(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        c0 = self.col(add_coords(obs))
        f = F.relu(self.n1(self.c1(c0)))
        h = torch.cat([c0, f], dim=1)
        s = self.film(h) + self.inter(h) + self.anchor(h) + self.tbias.view(1, -1, 1, 1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([s.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
