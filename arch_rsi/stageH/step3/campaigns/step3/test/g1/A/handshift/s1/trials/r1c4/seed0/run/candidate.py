import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0003, "n_envs": 64, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.003, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

STEM = 64
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
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(cin, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, W, 1)), nn.GroupNorm(8, W), nn.ReLU())
        self.c0 = init_layer(nn.Conv2d(W, W, 3, padding=1))
        self.n0 = nn.GroupNorm(8, W)
        self.res = Res(W)
        self.proj = init_layer(nn.Conv2d(W, PROJ, 1), 1.0)
        self.raw = init_layer(nn.Conv2d(cin, RAW, 1), 1.0)
        cat = PROJ + RAW
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(3 * cat, HID, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(cat, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.gproj = init_layer(nn.Conv2d(cin, GRAW, 1))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * W + 2 * GRAW, GLOB)), nn.ReLU())
        self.dense = init_layer(nn.Linear(GLOB, spec.n_place), 0.01)
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        x = add_coords(obs)
        s = self.stem(x)
        h = F.relu(self.n0(self.c0(s)))
        f = self.res(h)
        g = F.relu(self.gproj(x))
        gf = torch.cat([f.mean((2, 3)), f.amax((2, 3)), g.mean((2, 3)), g.amax((2, 3))], dim=1)
        gh = self.glob(gf)
        p = torch.cat([self.proj(f), self.raw(x)], dim=1)
        a = self.anchor(p)
        outs = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = fp_pool(p, dy, dx)
            hh = F.relu(self.mix[i](torch.cat([mx, mn, av], dim=1)))
            outs.append(self.outs[i](hh) + a + self.tbias[i])
        place = torch.cat(outs, dim=1).flatten(1) + self.dense(gh)
        logits = torch.cat([place, self.done_head(gh)], dim=1)
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
