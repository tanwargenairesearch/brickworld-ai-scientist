import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "ppo_epochs": 2, "n_minibatches": 16, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.3}

STEM = 64
W = 48
P = 32
HID = 32
K = 4


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
        SK = 2 * Z + 4
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.GroupNorm(8, STEM), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, W, spec.Y, spec.X) * 0.02)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(8, W)
        self.skip = nn.Conv2d(cin, SK, 1)
        self.lay = nn.Conv2d(W + SK + cin, 2 * Z, 1)
        self.proj = nn.Conv2d(W + SK, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P + 4 * Z, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.anchor = nn.Conv2d(W + SK, 1, 1)
        self.type_bias = nn.Parameter(torch.zeros(len(self.fps)))
        self.rproj = nn.Conv2d(W + SK, K, 1)
        self.remap = nn.Linear(K * spec.Y * spec.X, spec.n_place)
        self.glob = nn.Sequential(nn.Linear(2 * (W + SK) + 2 * Z, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self)
        for o in self.outs:
            small_init(o, 0.01)
        small_init(self.anchor, 0.01)
        small_init(self.done_head, 0.01)
        small_init(self.value_head, 1.0)
        nn.init.zeros_(self.remap.weight)
        nn.init.zeros_(self.remap.bias)

    def forward(self, obs, mask):
        Z = self.Z
        x0 = add_coords(obs)
        s = self.stem(x0) + self.pos
        h = F.relu(self.n1(self.c1(s)))
        f = F.relu(s + self.n2(self.c2(h)))
        sk = F.relu(self.skip(x0))
        fs = torch.cat([f, sk], dim=1)
        lz = torch.sigmoid(self.lay(torch.cat([fs, x0], dim=1)))
        occ = lz[:, :Z]
        miss = lz[:, Z:]
        p = self.proj(fs)
        a = self.anchor(fs)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            om = fmax(occ, dy, dx)
            z = torch.cat([fmax(p, dy, dx), fmin(p, dy, dx), favg(p, dy, dx),
                           om, revcummax(om), fmin(miss, dy, dx), fmax(miss, dy, dx)], dim=1)
            hh = F.relu(self.mix[i](z))
            res.append(self.outs[i](hh) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        place = place + self.remap(F.relu(self.rproj(fs)).flatten(1))
        g = self.glob(torch.cat([fs.mean((2, 3)), fs.amax((2, 3)), lz.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
