import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.003, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.995, "gae_lambda": 0.95, "clip_eps": 0.1}

COLW = 96
MID = 64
W = 40
PROJ = 24
RAW = 12
HID = 24
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


def done_stats(maps, fps):
    vals = []
    for m, (dy, dx) in zip(maps, fps):
        y, x = m.shape[2], m.shape[3]
        vals.append(m[:, 0, :max(1, y - dy + 1), :max(1, x - dx + 1)].flatten(1))
    v = torch.cat(vals, dim=1).detach()
    lse = torch.logsumexp(v, dim=1) - math.log(v.shape[1])
    return torch.stack([v.amax(1), lse, v.mean(1)], dim=1)


class AndHead(nn.Module):
    def __init__(self, cat, hid, fps):
        super().__init__()
        self.fps = fps
        self.a = nn.ModuleList([init_layer(nn.Conv2d(cat, hid, 1)) for _ in fps])
        self.c = nn.ModuleList([init_layer(nn.Conv2d(3 * cat, hid, 1)) for _ in fps])
        self.m = nn.ModuleList([init_layer(nn.Conv2d(hid, hid, 1)) for _ in fps])
        self.o = nn.ModuleList([init_layer(nn.Conv2d(hid, 1, 1), 0.01) for _ in fps])
        self.po = nn.ModuleList([init_layer(nn.Conv2d(hid, 1, 1), 0.01) for _ in fps])

    def forward(self, p):
        y, x = p.shape[2], p.shape[3]
        maps = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            ctx = self.c[i](torch.cat([mx, mn, av], dim=1))
            qa = F.pad(self.a[i](p), (0, dx - 1, 0, dy - 1))
            tot = self.po[i](F.relu(ctx))
            for oy in range(dy):
                for ox in range(dx):
                    h = F.relu(qa[:, :, oy:oy + y, ox:ox + x] + ctx)
                    h = F.relu(self.m[i](h))
                    tot = tot + F.logsigmoid(self.o[i](h)) + math.log(2.0)
            maps.append(tot)
        return maps


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col = nn.Sequential(
            init_layer(nn.Conv2d(cin, COLW, 1)), nn.GroupNorm(8, COLW), nn.ReLU(),
            init_layer(nn.Conv2d(COLW, COLW, 1)), nn.GroupNorm(8, COLW), nn.ReLU(),
            init_layer(nn.Conv2d(COLW, MID, 1)), nn.GroupNorm(8, MID), nn.ReLU())
        self.c1 = init_layer(nn.Conv2d(MID, W, 3, padding=1))
        self.n1 = nn.GroupNorm(8, W)
        self.proj = init_layer(nn.Conv2d(W + MID, PROJ, 1), 1.0)
        self.raw = init_layer(nn.Conv2d(cin, RAW, 1), 1.0)
        cat = PROJ + RAW
        self.head = AndHead(cat, HID, self.fps)
        self.anchor = init_layer(nn.Conv2d(cat, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.abias = nn.Parameter(torch.zeros(spec.n_actions))
        self.gproj = init_layer(nn.Conv2d(cin, GRAW, 1))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * W + 2 * MID + 2 * GRAW, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.done_stat = zero_layer(nn.Linear(3, 1))
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        x = add_coords(obs)
        c = self.col(x)
        f = F.relu(self.n1(self.c1(c)))
        p = torch.cat([self.proj(torch.cat([f, c], dim=1)), self.raw(x)], dim=1)
        a = self.anchor(p)
        maps = [m + a + self.tbias[i] for i, m in enumerate(self.head(p))]
        place = torch.cat(maps, dim=1).flatten(1)
        g = F.relu(self.gproj(x))
        gf = torch.cat([f.mean((2, 3)), f.amax((2, 3)), c.mean((2, 3)), c.amax((2, 3)), g.mean((2, 3)), g.amax((2, 3))], dim=1)
        gh = self.glob(gf)
        done = self.done_head(gh) + self.done_stat(done_stats(maps, self.fps))
        logits = torch.cat([place, done], dim=1) + self.abias
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
