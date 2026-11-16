import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 2, 'ent_coef': 0.01, 'gamma': 0.97, 'gae_lambda': 0.95, 'clip_eps': 0.2}

STEM = 64
W = 32
TOK = 64
SKIP = 16
PROJ = 32
HID = 32
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


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


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
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.nt = len(self.fps)
        self.stem = nn.Sequential(nn.Conv2d(cin + 2, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, W, 1))
        self.trunk = nn.Sequential(Block(W), TokenMix(W, n, TOK), Block(W), TokenMix(W, n, TOK))
        self.nf = nn.GroupNorm(4, W)
        self.skip = nn.Conv2d(cin, SKIP, 1)
        self.proj = nn.Conv2d(W + SKIP, PROJ, 1)
        self.anchor = nn.Conv2d(W + SKIP, 1, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * (W + SKIP), G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.type_bias = nn.Linear(G, self.nt)
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Sequential(nn.Linear(G, G), nn.ReLU(), nn.Linear(G, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                orth(m, math.sqrt(2.0))
        for blk in self.trunk:
            if isinstance(blk, TokenMix):
                orth(blk.l2, 1.0)
        for m in self.outs:
            orth(m, 0.01)
        orth(self.anchor, 0.01)
        orth(self.type_bias, 0.01)
        orth(self.done_head, 0.01)
        orth(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        x = self.trunk(self.stem(torch.cat([obs, coord_channels(obs)], dim=1)))
        f = torch.cat([F.relu(self.nf(x)), F.relu(self.skip(obs))], dim=1)
        p = self.proj(f)
        anc = self.anchor(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        tb = self.type_bias(g)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = fp_pool(p, dy, dx)
            h = F.relu(self.mix[i](torch.cat([mx, mn, av], dim=1)))
            res.append(self.outs[i](h) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
