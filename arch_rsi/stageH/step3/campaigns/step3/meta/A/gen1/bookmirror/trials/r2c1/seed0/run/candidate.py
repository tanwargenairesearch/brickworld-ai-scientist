import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95}

D = 64
HD = 96
NB = 3
SK = 16
P = 32
HID = 32


def add_coords(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def fmax(p, dy, dx):
    if dy * dx == 1:
        return p
    return F.max_pool2d(F.pad(p, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)


def fmin(p, dy, dx):
    return -fmax(-p, dy, dx)


def favg(p, dy, dx):
    if dy * dx == 1:
        return p
    return F.avg_pool2d(F.pad(p, (0, dx - 1, 0, dy - 1)), (dy, dx), stride=1)


def ortho_init(net, gain):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def small_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class MixBlock(nn.Module):
    def __init__(self, n, d, hd):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.tok = nn.Linear(n, n)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hd)
        self.fc2 = nn.Linear(hd, d)

    def forward(self, h):
        t = self.tok(self.ln1(h).transpose(1, 2)).transpose(1, 2)
        h = h + F.gelu(t)
        h = h + self.fc2(F.gelu(self.fc1(self.ln2(h))))
        return h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = spec.Y
        self.X = spec.X
        n = spec.Y * spec.X
        cin = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.embed = nn.Linear(cin, D)
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.blocks = nn.ModuleList([MixBlock(n, D, HD) for i in range(NB)])
        self.lnf = nn.LayerNorm(D)
        self.skip = nn.Linear(cin, SK)
        self.proj = nn.Linear(D + SK, P)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for f in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for f in self.fps])
        self.anchor = nn.Linear(D + SK, 1)
        self.type_bias = nn.Parameter(torch.zeros(len(self.fps)))
        self.omix = nn.Linear(n, n)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self, 1.0)
        for o in self.outs:
            small_init(o, 0.01)
        small_init(self.anchor, 0.01)
        small_init(self.done_head, 0.01)
        small_init(self.value_head, 1.0)
        nn.init.zeros_(self.omix.weight)
        nn.init.zeros_(self.omix.bias)

    def forward(self, obs, mask):
        b = obs.shape[0]
        y = self.Y
        x = self.X
        xt = add_coords(obs).flatten(2).transpose(1, 2)
        h = self.embed(xt) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.lnf(h)
        sk = F.relu(self.skip(xt))
        hs = torch.cat([h, sk], dim=2)
        p = self.proj(hs).transpose(1, 2).reshape(b, P, y, x)
        a = self.anchor(hs).transpose(1, 2).reshape(b, 1, y, x)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat([fmax(p, dy, dx), fmin(p, dy, dx), favg(p, dy, dx)], dim=1)
            hh = F.relu(self.mix[i](z))
            res.append(self.outs[i](hh) + a + self.type_bias[i])
        loc = torch.cat(res, dim=1).flatten(2)
        place = (loc + self.omix(loc)).flatten(1)
        g = self.glob(torch.cat([h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
