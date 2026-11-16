import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.003, "n_envs": 64, "rollout_len": 64, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.995, "gae_lambda": 0.99, "clip_eps": 0.3}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
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


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = init_layer(nn.Conv2d(c, proj, 1), 1.0)
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(3 * proj, hid, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(hid, 1, 1), 0.01) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        self.stem = nn.Sequential(
            init_layer(nn.Conv2d(cin, STEM, 1)), nn.ReLU(),
            init_layer(nn.Conv2d(STEM, COL, 1)), nn.GroupNorm(8, COL), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.gproj = init_layer(nn.Conv2d(cin, GRAW, 1))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * WIDTH + 2 * GRAW, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        x = add_coords(obs)
        s = self.stem(x)
        h = F.relu(self.n1(self.c1(s)))
        f = F.relu(h + self.n2(self.c2(h)))
        place = self.head(f)
        g = F.relu(self.gproj(x))
        gf = torch.cat([f.mean((2, 3)), f.amax((2, 3)), g.mean((2, 3)), g.amax((2, 3))], dim=1)
        gh = self.glob(gf)
        logits = torch.cat([place, self.done_head(gh)], dim=1)
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
