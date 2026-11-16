import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 64, "n_minibatches": 32,
          "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95,
          "clip_eps": 0.25}

STEM = 64
WIDTH = 40
GROUPS = 8
DEPTH = 2
SKIP = 16
LP = 16
LH = 24
GL = 96


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def foot_pool(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


def add_coords(obs):
    B, _, Y, X = obs.shape
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([obs, ys, xs], dim=1)


class Block(nn.Module):
    def __init__(self, c, groups):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.norm = nn.GroupNorm(groups, c)

    def forward(self, x):
        return F.relu(x + self.norm(self.conv(x)))


class LayerHead(nn.Module):
    def __init__(self, spec, ctrunk, p, hid):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        n = len(self.fps)
        self.inp = nn.Conv2d(12, p, 1)
        self.tproj = nn.Conv2d(ctrunk, p, 1)
        self.c2 = nn.Conv2d(p, p, 1)
        self.mix = nn.Conv2d(3 * p, hid, 1)
        self.temb = nn.Parameter(torch.zeros(n, hid, 1, 1))
        self.out_shared = nn.Conv2d(hid, 1, 1)
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        self.tbias = nn.Parameter(torch.zeros(n))

    def init_out(self):
        ortho(self.out_shared, 0.01)
        ortho(self.outs, 0.01)

    def forward(self, obs, f):
        B, _, Y, X = obs.shape
        Z = self.Z
        o = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(o, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(2).expand(B, 3, Z, Y, X)
        lp = (torch.arange(Z, device=obs.device, dtype=obs.dtype) / Z).view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        li = torch.cat([o, below, gl, lp], dim=1)
        li = li.permute(0, 2, 1, 3, 4).reshape(B * Z, 12, Y, X)
        tp = self.tproj(f)
        P = tp.shape[1]
        tp = tp.unsqueeze(1).expand(B, Z, P, Y, X).reshape(B * Z, P, Y, X)
        u = F.relu(self.inp(li) + tp)
        u = F.relu(self.c2(u))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = foot_pool(u, dy, dx)
            h = F.relu(self.mix(torch.cat([mx, mn, av], dim=1)) + self.temb[i])
            res.append(self.out_shared(h) + self.outs[i](h) + self.tbias[i])
        s = torch.cat(res, dim=1).view(B, Z, len(self.fps), Y, X)
        return (torch.logsumexp(s, dim=1) - math.log(Z)).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(c + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.GroupNorm(GROUPS, WIDTH), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(WIDTH, GROUPS) for _ in range(DEPTH)])
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.head = LayerHead(spec, WIDTH + SKIP, LP, LH)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * SKIP, GL), nn.ReLU())
        self.dense = nn.Linear(GL, spec.n_place)
        self.done_head = nn.Linear(GL, 1)
        self.value_head = nn.Sequential(nn.Linear(GL, GL), nn.ReLU(), nn.Linear(GL, 1))
        ortho(self, math.sqrt(2.0))
        self.head.init_out()
        ortho(self.dense, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        f = self.blocks(self.stem(add_coords(obs)))
        raw = F.relu(self.skip(obs))
        fr = torch.cat([f, raw], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                                 raw.mean((2, 3)), raw.amax((2, 3))], dim=1))
        place = self.head(obs, fr) + self.dense(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
