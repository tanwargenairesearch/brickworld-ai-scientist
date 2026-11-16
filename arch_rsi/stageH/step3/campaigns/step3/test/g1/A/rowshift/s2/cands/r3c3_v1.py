import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95, "clip_eps": 0.2}

H1 = 192
H2 = 128
NRB = 1
CTX = 24
RAW = 16
PROJ = 48
HID = 48
HID2 = 48
DF = 24
GH = 192


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
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        g = math.sqrt(2.0)
        self.s1 = init_layer(nn.Conv2d(c_in, H1, 1), g)
        self.n1 = ChanLN(H1)
        self.s2 = init_layer(nn.Conv2d(H1, H2, 1), g)
        self.n2 = ChanLN(H2)
        self.rc = nn.ModuleList([init_layer(nn.Conv2d(H2, H2, 1), g) for _ in range(NRB)])
        self.rn = nn.ModuleList([ChanLN(H2) for _ in range(NRB)])
        self.ctx = init_layer(nn.Conv2d(H2, CTX, 3, padding=1), g)
        self.raw = init_layer(nn.Conv2d(c_in, RAW, 1), 1.0)
        fd = H2 + CTX + RAW
        self.proj = init_layer(nn.Conv2d(fd, PROJ, 1), 1.0)
        self.mix1 = nn.ModuleList([init_layer(nn.Conv2d(3 * PROJ, HID, 1), g) for _ in self.fps])
        self.mix2 = nn.ModuleList([init_layer(nn.Conv2d(HID, HID2, 1), g) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID2, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(fd, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.dfeat = init_layer(nn.Conv2d(fd, DF, 1), g)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * fd, GH), g), nn.ReLU())
        self.dmlp = nn.Sequential(init_layer(nn.Linear(2 * DF + GH, 32), g), nn.ReLU(),
                                  init_layer(nn.Linear(32, 1), 0.01))
        self.value = init_layer(nn.Linear(GH + 2 * DF, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.n1(self.s1(obs)))
        s = F.relu(self.n2(self.s2(s)))
        for conv, ln in zip(self.rc, self.rn):
            s = s + F.relu(ln(conv(s)))
        c = F.relu(self.ctx(s))
        f = torch.cat([s, c, self.raw(obs)], dim=1)
        p = self.proj(f)
        a = self.anchor(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            h = F.relu(self.mix1[i](torch.cat(fp_pool(p, dy, dx), dim=1)))
            h = F.relu(self.mix2[i](h))
            res.append(self.outs[i](h) + a + self.tbias[i])
        place = torch.cat(res, dim=1).flatten(1)
        d = F.relu(self.dfeat(f))
        ds = torch.cat([d.amax((2, 3)), d.mean((2, 3))], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.dmlp(torch.cat([ds, g], dim=1))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value(torch.cat([g, ds], dim=1)).squeeze(1)


def build_policy(spec):
    return Net(spec)
