import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 5e-4, "n_envs": 256, "rollout_len": 32, "n_minibatches": 64,
          "ppo_epochs": 3, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.02}

LZ = 16
STEM = 64
WIDTH = 40
HID = 32
GLOB = 64
GROUPS = 8
DILS = [1, 2]


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def coord_planes(obs):
    b, _, h, w = obs.shape
    ys = torch.linspace(-1.0, 1.0, h, device=obs.device, dtype=obs.dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=obs.device, dtype=obs.dtype)
    gy = ys.view(1, 1, h, 1).expand(b, 1, h, w)
    gx = xs.view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([gy, gx], dim=1)


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
        self.Z = int(spec.Z)
        self.lz1 = init_layer(nn.Conv2d(4, LZ, (3, 1), padding=(1, 0)))
        self.lz2 = init_layer(nn.Conv2d(LZ, LZ, (3, 1), padding=(1, 0)))
        rest = spec.in_channels - 4 * self.Z
        self.mixz = init_layer(nn.Conv2d(LZ * self.Z + rest + 2, STEM, 1))
        self.stem2 = init_layer(nn.Conv2d(STEM, WIDTH, 1))
        self.blocks = nn.ModuleList([Block(WIDTH, d) for d in DILS])
        self.nf = nn.GroupNorm(GROUPS, WIDTH)
        self.head = CenterHead(WIDTH, spec.footprints)
        self.att = init_layer(nn.Conv2d(WIDTH, 1, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(3 * WIDTH, GLOB)), nn.ReLU(),
                                  init_layer(nn.Linear(GLOB, GLOB)), nn.ReLU())
        self.type_bias = init_layer(nn.Linear(GLOB, spec.n_types), 0.01)
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.vhid = nn.Sequential(init_layer(nn.Linear(3 * WIDTH, GLOB)), nn.ReLU())
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        b, _, h, w = obs.shape
        z = self.Z
        v = obs[:, :4 * z].reshape(b, 4, z, h * w)
        l1 = F.relu(self.lz1(v))
        l2 = F.relu(self.lz2(l1)) + l1
        lf = l2.reshape(b, LZ * z, h, w)
        inp = torch.cat([lf, obs[:, 4 * z:], coord_planes(obs)], dim=1)
        x = self.stem2(F.relu(self.mixz(inp)))
        for blk in self.blocks:
            x = blk(x)
        f = F.relu(self.nf(x))
        wt = torch.softmax(self.att(f).flatten(1), dim=1)
        a = (f.flatten(2) * wt.unsqueeze(1)).sum(2)
        p = torch.cat([f.mean((2, 3)), f.amax((2, 3)), a], dim=1)
        g = self.glob(p)
        place = self.head(f) + self.type_bias(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        value = self.value_head(self.vhid(p)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
