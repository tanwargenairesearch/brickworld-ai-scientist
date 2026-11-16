import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ppo_epochs": 3, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

MUL = 48
LIN = 32
W = 40
PROJ = 32
RAW = 16
HID = 32
GRAW = 16
GLOB = 64


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def zero_layer(m):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


def done_stats(maps, fps):
    vals = []
    for m, (dy, dx) in zip(maps, fps):
        y, x = m.shape[2], m.shape[3]
        vals.append(m[:, 0, :max(1, y - dy + 1), :max(1, x - dx + 1)].flatten(1))
    v = torch.cat(vals, dim=1).detach()
    lse = torch.logsumexp(v, dim=1) - math.log(v.shape[1])
    return torch.stack([v.amax(1), lse, v.mean(1)], dim=1)


class MulStem(nn.Module):
    def __init__(self, cin, m, lin, out):
        super().__init__()
        self.a = init_layer(nn.Conv2d(cin, m, 1), 1.0)
        self.b = init_layer(nn.Conv2d(cin, m, 1), 1.0)
        self.l = init_layer(nn.Conv2d(cin, lin, 1))
        self.n = nn.GroupNorm(8, m + lin)
        self.o = init_layer(nn.Conv2d(m + lin, out, 1))
        self.no = nn.GroupNorm(8, out)

    def forward(self, x):
        h = torch.cat([self.a(x) * self.b(x), F.relu(self.l(x))], dim=1)
        h = F.relu(self.n(h))
        return F.relu(self.no(self.o(h)))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = init_layer(nn.Conv2d(c, c, 3, padding=1))
        self.n1 = nn.GroupNorm(8, c)
        self.c2 = init_layer(nn.Conv2d(c, c, 3, padding=1))
        self.n2 = nn.GroupNorm(8, c)

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        return F.relu(x + self.n2(self.c2(h)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = MulStem(cin, MUL, LIN, W)
        self.c0 = init_layer(nn.Conv2d(W, W, 3, padding=1))
        self.n0 = nn.GroupNorm(8, W)
        self.res = Res(W)
        self.proj = init_layer(nn.Conv2d(2 * W, PROJ, 1), 1.0)
        self.raw = init_layer(nn.Conv2d(cin, RAW, 1), 1.0)
        cat = PROJ + RAW
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(3 * cat, HID, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(cat, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.gproj = init_layer(nn.Conv2d(cin, GRAW, 1))
        self.glob = nn.Sequential(init_layer(nn.Linear(4 * W + 2 * GRAW, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.done_stat = zero_layer(nn.Linear(3, 1))
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        x = add_coords(obs)
        s = self.stem(x)
        h = F.relu(self.n0(self.c0(s)))
        f = self.res(h)
        sf = torch.cat([f, s], dim=1)
        p = torch.cat([self.proj(sf), self.raw(x)], dim=1)
        a = self.anchor(p)
        maps = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = fp_pool(p, dy, dx)
            hh = F.relu(self.mix[i](torch.cat([mx, mn, av], dim=1)))
            maps.append(self.outs[i](hh) + a + self.tbias[i])
        place = torch.cat(maps, dim=1).flatten(1)
        g = F.relu(self.gproj(x))
        gf = torch.cat([sf.mean((2, 3)), sf.amax((2, 3)), g.mean((2, 3)), g.amax((2, 3))], dim=1)
        gh = self.glob(gf)
        done = self.done_head(gh) + self.done_stat(done_stats(maps, self.fps))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
