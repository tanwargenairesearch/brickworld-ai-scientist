import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.98, "gae_lambda": 0.95, "clip_eps": 0.2}

H1 = 96
H2 = 64
CTX = 24
RAW = 16
PROJ = 24
HID = 32
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
        self.Y, self.X = int(spec.Y), int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.nt = nt
        self.np = int(spec.n_place)
        c_in = spec.in_channels + nt
        g = math.sqrt(2.0)
        self.s1 = init_layer(nn.Conv2d(c_in, H1, 1), g)
        self.n1 = ChanLN(H1)
        self.s2 = init_layer(nn.Conv2d(H1, H2, 1), g)
        self.n2 = ChanLN(H2)
        self.s3 = init_layer(nn.Conv2d(H2, H2, 1), g)
        self.n3 = ChanLN(H2)
        self.ctx = init_layer(nn.Conv2d(H2, CTX, 3, padding=1), g)
        self.raw = init_layer(nn.Conv2d(c_in, RAW, 1), 1.0)
        fd = H2 + CTX + RAW
        self.proj = init_layer(nn.Conv2d(fd, PROJ, 1), 1.0)
        self.fconv = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx)), g) for dy, dx in self.fps])
        self.pmix = nn.ModuleList([init_layer(nn.Conv2d(3 * PROJ, HID, 1), g) for _ in self.fps])
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(HID, HID, 1), g) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(fd, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(nt))
        self.dfeat = init_layer(nn.Conv2d(fd, DF, 1), g)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * fd, GH), g), nn.ReLU())
        zin = GH + 2 * DF + 2 * nt * HID
        self.dmlp = nn.Sequential(init_layer(nn.Linear(zin, 48), g), nn.ReLU(),
                                  init_layer(nn.Linear(48, 1), 0.01))
        self.vmlp = nn.Sequential(init_layer(nn.Linear(zin, 64), g), nn.ReLU(),
                                  init_layer(nn.Linear(64, 1), 1.0))

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.np].float().reshape(B, self.nt, self.Y, self.X)
        x = torch.cat([obs, m], dim=1)
        s = F.relu(self.n1(self.s1(x)))
        s = F.relu(self.n2(self.s2(s)))
        s = s + F.relu(self.n3(self.s3(s)))
        c = F.relu(self.ctx(s))
        f = torch.cat([s, c, self.raw(x)], dim=1)
        p = self.proj(f)
        a = self.anchor(f)
        res = []
        hst = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            h = self.fconv[i](q) + self.pmix[i](torch.cat(fp_pool(p, dy, dx), dim=1))
            h = F.relu(h)
            h = F.relu(self.mix[i](h))
            hst.append(h.amax((2, 3)))
            hst.append(h.mean((2, 3)))
            res.append(self.outs[i](h) + a + self.tbias[i])
        place = torch.cat(res, dim=1).flatten(1)
        dd = F.relu(self.dfeat(f))
        ds = torch.cat([dd.amax((2, 3)), dd.mean((2, 3))], dim=1)
        gg = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        z = torch.cat([gg, ds] + hst, dim=1)
        done = self.dmlp(z)
        logits = torch.cat([place, done], dim=1)
        return logits, self.vmlp(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
