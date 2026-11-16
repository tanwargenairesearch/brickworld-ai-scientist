import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 7e-4, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64,
          "ppo_epochs": 4, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.003}

STEM = 64
WIDTH = 40
PROJ = 32
HID = 32
GLOB = 64
GROUPS = 8
DILS = [1, 2]


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b, _, h, w = obs.shape
    ys = torch.linspace(-1.0, 1.0, h, device=obs.device, dtype=obs.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=obs.device, dtype=obs.dtype)
    gy = ys.view(1, 1, h, 1).expand(b, 1, h, w)
    gx = xs.view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([obs, gy, gx], dim=1)


class Block(nn.Module):
    def __init__(self, c, d=1):
        super().__init__()
        self.n1 = nn.GroupNorm(GROUPS, c)
        self.c1 = init_layer(nn.Conv2d(c, c, 3, padding=d, dilation=d))
        self.n2 = nn.GroupNorm(GROUPS, c)
        self.c2 = init_layer(nn.Conv2d(c, c, 1), 1.0)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class PoolHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1))
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(2 * PROJ, HID, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn = p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            res.append(out(F.relu(mix(torch.cat([mx, mn], dim=1)))))
        return torch.cat(res, dim=1)


class CenterHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        k = max(max(int(a), int(b)) for a, b in footprints)
        ks = 2 * k + 1
        self.hid = init_layer(nn.Conv2d(c, HID, 3, padding=1))
        self.out = init_layer(nn.Conv2d(HID, len(footprints), ks, padding=ks // 2), 0.01)

    def forward(self, f):
        return self.out(F.relu(self.hid(f)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        cin = spec.in_channels + 2 + self.nt
        self.stem1 = init_layer(nn.Conv2d(cin, STEM, 1))
        self.stem2 = init_layer(nn.Conv2d(STEM, WIDTH, 1))
        self.blocks = nn.ModuleList([Block(WIDTH, d) for d in DILS])
        self.nf = nn.GroupNorm(GROUPS, WIDTH)
        self.phead = PoolHead(WIDTH, spec.footprints)
        self.chead = CenterHead(WIDTH, spec.footprints)
        self.att = init_layer(nn.Conv2d(WIDTH, 1, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(3 * WIDTH, GLOB)), nn.ReLU(),
                                  init_layer(nn.Linear(GLOB, GLOB)), nn.ReLU())
        self.type_bias = init_layer(nn.Linear(GLOB, spec.n_types), 0.01)
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.vhid = nn.Sequential(init_layer(nn.Linear(3 * WIDTH, GLOB)), nn.ReLU())
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        b = obs.shape[0]
        mp = mask[:, :self.npl].to(obs.dtype).reshape(b, self.nt, self.Y, self.X)
        inp = torch.cat([add_coords(obs), mp], dim=1)
        x = self.stem2(F.relu(self.stem1(inp)))
        for blk in self.blocks:
            x = blk(x)
        f = F.relu(self.nf(x))
        w = torch.softmax(self.att(f).flatten(1), dim=1)
        a = (f.flatten(2) * w.unsqueeze(1)).sum(2)
        p = torch.cat([f.mean((2, 3)), f.amax((2, 3)), a], dim=1)
        g = self.glob(p)
        place = self.phead(f) + self.chead(f) + self.type_bias(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        value = self.value_head(self.vhid(p)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
