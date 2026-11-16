import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1.5e-3, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32,
          "ppo_epochs": 1, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

STEM = 64
WIDTH = 32
ITERS = 4
HID = 32
GLOB = 64
GROUPS = 8


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
        cin = spec.in_channels + 2
        self.stem1 = init_layer(nn.Conv2d(cin, STEM, 1))
        self.stem2 = init_layer(nn.Conv2d(STEM, WIDTH, 1))
        self.n1 = nn.ModuleList([nn.GroupNorm(GROUPS, WIDTH) for _ in range(ITERS)])
        self.n2 = nn.ModuleList([nn.GroupNorm(GROUPS, WIDTH) for _ in range(ITERS)])
        self.c1 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.c2 = init_layer(nn.Conv2d(WIDTH, WIDTH, 1), 1.0)
        self.nf = nn.GroupNorm(GROUPS, WIDTH)
        self.head = CenterHead(2 * WIDTH, spec.footprints)
        self.att = init_layer(nn.Conv2d(2 * WIDTH, 1, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(6 * WIDTH, GLOB)), nn.ReLU(),
                                  init_layer(nn.Linear(GLOB, GLOB)), nn.ReLU())
        self.type_bias = init_layer(nn.Linear(GLOB, spec.n_types), 0.01)
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.vhid = nn.Sequential(init_layer(nn.Linear(6 * WIDTH, GLOB)), nn.ReLU())
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        e = self.stem2(F.relu(self.stem1(add_coords(obs))))
        x = e
        for i in range(ITERS):
            h = self.c1(F.relu(self.n1[i](x)))
            h = self.c2(F.relu(self.n2[i](h)))
            x = x + h
        f = torch.cat([F.relu(self.nf(x)), F.relu(e)], dim=1)
        w = torch.softmax(self.att(f).flatten(1), dim=1)
        a = (f.flatten(2) * w.unsqueeze(1)).sum(2)
        p = torch.cat([f.mean((2, 3)), f.amax((2, 3)), a], dim=1)
        g = self.glob(p)
        place = self.head(f) + self.type_bias(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        value = self.value_head(self.vhid(p)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
