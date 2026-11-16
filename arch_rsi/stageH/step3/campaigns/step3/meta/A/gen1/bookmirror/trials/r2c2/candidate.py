import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95}

E = 48
W = 48
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


def grid_transforms(t, square):
    out = [torch.flip(t, [3]), torch.flip(t, [2]), torch.flip(t, [2, 3])]
    if square:
        u = t.transpose(2, 3)
        out = out + [u, torch.flip(u, [3]), torch.flip(u, [2]), torch.flip(u, [2, 3])]
    return out


def shift(t, dy, dx):
    y, x = t.shape[2], t.shape[3]
    q = F.pad(t, (2, 2, 2, 2))
    return q[:, :, 2 + dy:2 + dy + y, 2 + dx:2 + dx + x]


def shift_transforms(t):
    return [shift(t, 1, 0), shift(t, -1, 0), shift(t, 0, 1), shift(t, 0, -1)]


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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.square = spec.Y == spec.X
        k1 = 7 if self.square else 3
        k2 = k1 + 4
        self.enc = nn.Sequential(nn.Conv2d(cin, E, 1), nn.GroupNorm(8, E), nn.ReLU())
        self.align = nn.Parameter(torch.zeros(k1, E))
        self.stem = nn.Sequential(nn.Conv2d(2 * E, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = nn.GroupNorm(8, W)
        self.skip = nn.Conv2d(cin, SK, 1)
        self.proj = nn.Conv2d(W + SK, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for f in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for f in self.fps])
        self.anchor = nn.Conv2d(W + SK, 1, 1)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.owt = nn.Parameter(torch.zeros(k2, T))
        self.tmix = nn.Parameter(torch.zeros(T, T))
        self.glob = nn.Sequential(nn.Linear(2 * (W + SK), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self, math.sqrt(2.0))
        for o in self.outs:
            small_init(o, 0.01)
        small_init(self.anchor, 0.01)
        small_init(self.done_head, 0.01)
        small_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        x0 = add_coords(obs)
        e = self.enc(x0)
        al = torch.zeros_like(e)
        for k, t in enumerate(grid_transforms(e, self.square)):
            al = al + self.align[k].view(1, -1, 1, 1) * t
        s = self.stem(torch.cat([e, al], dim=1))
        h = F.relu(self.n1(self.c1(s)))
        f = F.relu(s + self.n2(self.c2(h)))
        sk = F.relu(self.skip(x0))
        fs = torch.cat([f, sk], dim=1)
        p = self.proj(fs)
        a = self.anchor(fs)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat([fmax(p, dy, dx), fmin(p, dy, dx), favg(p, dy, dx)], dim=1)
            hh = F.relu(self.mix[i](z))
            res.append(self.outs[i](hh) + a + self.type_bias[i])
        loc = torch.cat(res, dim=1)
        place = loc + torch.einsum("ts,bsyx->btyx", self.tmix, loc)
        for k, t in enumerate(grid_transforms(loc, self.square) + shift_transforms(loc)):
            place = place + self.owt[k].view(1, -1, 1, 1) * t
        place = place.flatten(1)
        g = self.glob(torch.cat([fs.mean((2, 3)), fs.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
