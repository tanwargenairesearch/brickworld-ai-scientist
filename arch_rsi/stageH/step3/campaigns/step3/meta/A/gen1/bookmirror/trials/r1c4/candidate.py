import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 512, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
W = 48
P = 24
HID = 48


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


def revcummax(t):
    return torch.flip(torch.cummax(torch.flip(t, [1]), dim=1)[0], [1])


def ortho_init(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def small_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        Z = spec.Z
        self.Z = Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.GroupNorm(1, STEM), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.GroupNorm(1, W), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(1, W)
        self.lay = nn.Conv2d(W + cin, 3 * Z, 1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P + 5 * Z, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(W, 1, 1)
        self.type_bias = nn.Parameter(torch.zeros(len(self.fps)))
        self.glob = nn.Sequential(nn.Linear(2 * W + 3 * Z, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self)
        for o in self.outs:
            small_init(o, 0.01)
        small_init(self.anchor, 0.01)
        small_init(self.done_head, 0.01)
        small_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        Z = self.Z
        x0 = add_coords(obs)
        s = self.stem(x0)
        f = F.relu(s + self.n1(self.c1(s)))
        lz = torch.sigmoid(self.lay(torch.cat([f, x0], dim=1)))
        occ = lz[:, :Z]
        miss = lz[:, Z:2 * Z]
        tgt = lz[:, 2 * Z:]
        p = self.proj(f)
        a = self.anchor(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            om = fmax(occ, dy, dx)
            z = torch.cat([fmax(p, dy, dx), fmin(p, dy, dx), favg(p, dy, dx),
                           om, revcummax(om), fmin(miss, dy, dx), fmax(miss, dy, dx),
                           fmax(tgt, dy, dx)], dim=1)
            hh = F.relu(self.mix[i](z))
            res.append(self.outs[i](hh) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), lz.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
