import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.002, 'n_envs': 256, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 1, 'ent_coef': 0.01, 'gamma': 0.98, 'gae_lambda': 0.95, 'clip_eps': 0.2}

STEM = 64
W = 32
TOK = 64
SKIP = 16
PROJ = 16
MIX = 48
PT = 8
KR = 4
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


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class TokenMix(nn.Module):
    def __init__(self, c, n, hid):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.l1 = nn.Linear(n, hid)
        self.l2 = nn.Linear(hid, n)

    def forward(self, x):
        b, c, y, xx = x.shape
        h = self.norm(x).reshape(b, c, y * xx)
        h = self.l2(F.relu(self.l1(h)))
        return x + h.reshape(b, c, y, xx)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        n = int(spec.Y) * int(spec.X)
        self.n = n
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        t = len(self.fps)
        self.nt = t
        self.shapes = sorted(set([f for f in self.fps if f[0] * f[1] > 1]))
        self.stem = nn.Sequential(nn.Conv2d(cin + 2, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, W, 1))
        self.tmix = TokenMix(W, n, TOK)
        self.trunk = nn.Sequential(Block(W), self.tmix, Block(W))
        self.nf = nn.GroupNorm(4, W)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.proj = nn.Conv2d(W + SKIP, PROJ, 1)
        self.mix = nn.Conv2d(PROJ * (1 + 2 * len(self.shapes)), MIX, 1)
        self.tconv = nn.Conv2d(MIX, t * PT, 1)
        self.g5 = nn.Conv2d(t * PT, t * PT, 5, padding=2, groups=t)
        self.gout = nn.Conv2d(t * PT, t, 1, groups=t)
        self.rfeat = nn.Conv2d(t * PT, t * KR, 1, groups=t)
        self.remap = nn.ModuleList([nn.Linear(KR * n, n) for _ in self.fps])
        self.abias = nn.Parameter(torch.zeros(1, spec.n_place))
        self.glob = nn.Sequential(nn.Linear(2 * (W + SKIP), G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.type_bias = nn.Linear(G, t)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        orth(self.tmix.l2, 1.0)
        orth(self.gout, 0.01)
        for m in self.remap:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        orth(self.type_bias, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        n = self.n
        x = self.trunk(self.stem(torch.cat([obs, coord_channels(obs)], dim=1)))
        f = torch.cat([F.relu(self.nf(x)), F.relu(self.skip(obs))], dim=1)
        p = self.proj(f)
        feats = [p]
        for dy, dx in self.shapes:
            mx, mn = fp_pool2(p, dy, dx)
            feats.append(mx)
            feats.append(mn)
        u = F.relu(self.mix(torch.cat(feats, dim=1)))
        v = F.relu(self.tconv(u))
        v = F.relu(v + self.g5(v))
        o = self.gout(v).flatten(1)
        r = F.relu(self.rfeat(v)).reshape(B, self.nt, KR * n)
        rem = torch.stack([self.remap[k](r[:, k]) for k in range(self.nt)], dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        tb = self.type_bias(g).unsqueeze(2).expand(B, self.nt, n).reshape(B, -1)
        place = o + rem + tb + self.abias
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
