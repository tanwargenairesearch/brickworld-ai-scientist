import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 64, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95, "clip_eps": 0.2}

H1 = 96
H2 = 48
D1 = 64
RAW = 16
PROJ = 24
HID = 24
DF = 16
GH = 128


def init_layer(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return [p, p, p]
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return [mx, mn, av]


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c_in = spec.in_channels
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        g = math.sqrt(2.0)
        self.s1 = init_layer(nn.Conv2d(c_in, H1, 1), g)
        self.n1 = ChanLN(H1)
        self.s2 = init_layer(nn.Conv2d(H1, H2, 1), g)
        self.n2 = ChanLN(H2)
        self.d1 = init_layer(nn.Conv2d(H2, D1, 3, stride=2, padding=1), g)
        self.gd1 = nn.GroupNorm(8, D1)
        self.d2 = init_layer(nn.Conv2d(D1, D1, 3, padding=1), g)
        self.gd2 = nn.GroupNorm(8, D1)
        self.gfc = init_layer(nn.Linear(2 * D1, D1), 1.0)
        self.up = init_layer(nn.Conv2d(H2 + D1, H2, 1), g)
        self.nu = ChanLN(H2)
        self.raw = init_layer(nn.Conv2d(c_in, RAW, 1), 1.0)
        fd = 2 * H2 + RAW
        self.proj = init_layer(nn.Conv2d(fd, PROJ, 1), 1.0)
        self.fconv = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx)), g) for dy, dx in self.fps])
        self.win = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, 3, padding=1), g) for _ in self.fps])
        self.pmix = nn.ModuleList([init_layer(nn.Conv2d(3 * PROJ, HID, 1), g) for _ in self.fps])
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(HID, HID, 1), g) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(fd, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(nt))
        self.dfeat = init_layer(nn.Conv2d(fd, DF, 1), g)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * fd + 2 * D1, GH), g), nn.ReLU())
        zin = GH + 2 * DF + nt * HID
        self.dmlp = nn.Sequential(init_layer(nn.Linear(zin, 32), g), nn.ReLU(),
                                  init_layer(nn.Linear(32, 1), 0.01))
        self.value = init_layer(nn.Linear(zin, 1), 1.0)

    def forward(self, obs, mask):
        s = F.silu(self.n1(self.s1(obs)))
        s = F.silu(self.n2(self.s2(s)))
        d = F.silu(self.gd1(self.d1(s)))
        d = F.silu(self.gd2(self.d2(d)))
        gd = torch.cat([d.mean((2, 3)), d.amax((2, 3))], dim=1)
        d = d + self.gfc(gd)[:, :, None, None]
        u = F.interpolate(d, size=(self.Y, self.X), mode="nearest")
        u = F.silu(self.nu(self.up(torch.cat([s, u], dim=1))))
        f = torch.cat([s, u, self.raw(obs)], dim=1)
        p = self.proj(f)
        a = self.anchor(f)
        res = []
        hst = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            h = self.fconv[i](q) + self.win[i](p) + self.pmix[i](torch.cat(fp_pool(p, dy, dx), dim=1))
            h = F.relu(h)
            h = F.relu(self.mix[i](h))
            hst.append(h.amax((2, 3)))
            res.append(self.outs[i](h) + a + self.tbias[i])
        place = torch.cat(res, dim=1).flatten(1)
        dd = F.relu(self.dfeat(f))
        ds = torch.cat([dd.amax((2, 3)), dd.mean((2, 3))], dim=1)
        gg = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), gd], dim=1))
        z = torch.cat([gg, ds] + hst, dim=1)
        done = self.dmlp(z)
        logits = torch.cat([place, done], dim=1)
        return logits, self.value(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
