import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95}

E = 32
W = 48
SK = 16
P = 32
HID = 32
NB = 2
HD = 48


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
        Z = spec.Z
        self.Z = Z
        n = spec.Y * spec.X
        cin = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.square = spec.Y == spec.X
        k1 = 7 if self.square else 3
        k2 = k1 + 4
        self.enc = nn.Conv2d(Z + 3, E, 3, padding=1)
        self.aux = nn.Conv2d(2 * Z + 3, E, 1)
        self.align = nn.Parameter(torch.zeros(k1, E))
        self.stem = nn.Sequential(nn.Conv2d(6 * E, W, 1), nn.GroupNorm(8, W), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.n1 = nn.GroupNorm(8, W)
        self.pos = nn.Parameter(torch.randn(1, n, W) * 0.02)
        self.blocks = nn.ModuleList([MixBlock(n, W, HD) for i in range(NB)])
        self.lnf = nn.LayerNorm(W)
        self.skip = nn.Conv2d(cin, SK, 1)
        self.lay = nn.Conv2d(W + SK + cin, 2 * Z, 1)
        self.proj = nn.Conv2d(W + SK, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P + 4 * Z, HID, 1) for f in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for f in self.fps])
        self.anchor = nn.Conv2d(W + SK, 1, 1)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.owt = nn.Parameter(torch.zeros(k2, T))
        self.tmix = nn.Parameter(torch.zeros(T, T))
        self.glob = nn.Sequential(nn.Linear(2 * (W + SK) + 2 * Z, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho_init(self, math.sqrt(2.0))
        for blk in self.blocks:
            ortho_init(blk, 1.0)
        for o in self.outs:
            small_init(o, 0.01)
        small_init(self.anchor, 0.01)
        small_init(self.done_head, 0.01)
        small_init(self.value_head, 1.0)

    def forward(self, obs, mask):
        Z = self.Z
        b, y, x = obs.shape[0], obs.shape[2], obs.shape[3]
        x0 = add_coords(obs)
        co = x0[:, 4 * Z + 3:]
        bld = torch.cat([x0[:, :Z], x0[:, 4 * Z:4 * Z + 1], co], dim=1)
        tgt = torch.cat([x0[:, Z:2 * Z], x0[:, 4 * Z + 1:4 * Z + 2], co], dim=1)
        aux = torch.cat([x0[:, 2 * Z:4 * Z], x0[:, 4 * Z + 2:4 * Z + 3], co], dim=1)
        eb = self.enc(bld)
        et = self.enc(tgt)
        ea = self.aux(aux)
        al = torch.zeros_like(et)
        for k, t in enumerate(grid_transforms(et, self.square)):
            al = al + self.align[k].view(1, -1, 1, 1) * t
        s = self.stem(torch.cat([F.relu(eb), F.relu(et), et - eb, eb * et, F.relu(ea), al], dim=1))
        f = F.relu(s + self.n1(self.c1(s)))
        h = f.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            h = blk(h)
        f = self.lnf(h).transpose(1, 2).reshape(b, W, y, x)
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
        loc = torch.cat(res, dim=1)
        place = loc + torch.einsum("ts,bsyx->btyx", self.tmix, loc)
        for k, t in enumerate(grid_transforms(loc, self.square) + shift_transforms(loc)):
            place = place + self.owt[k].view(1, -1, 1, 1) * t
        place = place.flatten(1)
        g = self.glob(torch.cat([fs.mean((2, 3)), fs.amax((2, 3)), lz.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
