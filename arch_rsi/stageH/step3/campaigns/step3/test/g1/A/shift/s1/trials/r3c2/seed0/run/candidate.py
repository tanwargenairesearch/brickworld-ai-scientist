import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.98, 'gae_lambda': 0.95, 'clip_eps': 0.2}

STEM = 64
COL = 48
W = 40
SKIP = 16
PROJ = 16
MIX = 48
PT = 8
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


def fp_pool2(p, dy, dx):
    if dy * dx == 1:
        return p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    return F.max_pool2d(q, (dy, dx), stride=1), -F.max_pool2d(-q, (dy, dx), stride=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        t = len(self.fps)
        self.shapes = sorted(set([f for f in self.fps if f[0] * f[1] > 1]))
        self.stem = nn.Sequential(nn.Conv2d(cin + 2, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, W, 3, padding=1)
        self.n1 = nn.GroupNorm(4, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(4, W)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.proj = nn.Conv2d(W + SKIP, PROJ, 1)
        self.mix = nn.Conv2d(PROJ * (1 + 2 * len(self.shapes)), MIX, 1)
        self.tconv = nn.Conv2d(MIX, t * PT, 1)
        self.g5 = nn.Conv2d(t * PT, t * PT, 5, padding=2, groups=t)
        self.gout = nn.Conv2d(t * PT, t, 1, groups=t)
        self.abias = nn.Parameter(torch.zeros(1, spec.n_place))
        self.glob = nn.Sequential(nn.Linear(2 * (W + SKIP), G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.dense = nn.Linear(G, spec.n_place)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        orth(self.gout, 0.01)
        orth(self.dense, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        s = self.stem(torch.cat([obs, coord_channels(obs)], dim=1))
        x = F.relu(self.n1(self.c1(s)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([x, F.relu(self.skip(obs))], dim=1)
        p = self.proj(f)
        feats = [p]
        for dy, dx in self.shapes:
            mx, mn = fp_pool2(p, dy, dx)
            feats.append(mx)
            feats.append(mn)
        u = F.relu(self.mix(torch.cat(feats, dim=1)))
        v = F.relu(self.tconv(u))
        v = F.relu(v + self.g5(v))
        o = self.gout(v)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = o.flatten(1) + self.abias + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
