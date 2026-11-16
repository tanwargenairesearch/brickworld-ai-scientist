import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 48, "n_minibatches": 48,
          "ppo_epochs": 1, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
PROJ = 24
HID = 32
GLOB = 64
GROUPS = 8


def init_layer(m, gain=1.4142135623730951):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class ConvHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.fconv = nn.ModuleList([init_layer(nn.Conv2d(c, HID, (dy, dx))) for dy, dx in self.fps])
        self.proj = init_layer(nn.Conv2d(c, PROJ, 1))
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(2 * PROJ, HID, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(c, 1, 1), 0.01)

    def forward(self, f, tbias):
        p = self.proj(f)
        res = []
        for (dy, dx), fc, mix, out in zip(self.fps, self.fconv, self.mix, self.outs):
            a = fc(F.pad(f, (0, dx - 1, 0, dy - 1)))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(a + mix(torch.cat([mx, mn], dim=1)))
            res.append(out(h))
        inter = torch.cat(res, dim=1)
        place = inter + self.anchor(f) + tbias[:, :, None, None]
        return place.flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        cin = spec.in_channels
        self.stem1 = init_layer(nn.Conv2d(cin, STEM, 1))
        self.stem2 = init_layer(nn.Conv2d(STEM, COL, 1))
        self.n0 = nn.GroupNorm(GROUPS, COL)
        self.c0 = init_layer(nn.Conv2d(COL, WIDTH, 3, padding=1))
        self.n1 = nn.GroupNorm(GROUPS, WIDTH)
        self.c1 = init_layer(nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.nf = nn.GroupNorm(GROUPS, WIDTH)
        self.skip = init_layer(nn.Conv2d(cin, SKIP, 1))
        fc = WIDTH + SKIP
        self.head = ConvHead(fc, spec.footprints)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * fc, GLOB)), nn.ReLU(),
                                  init_layer(nn.Linear(GLOB, GLOB)), nn.ReLU())
        self.type_bias = init_layer(nn.Linear(GLOB, spec.n_types), 0.01)
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        s = F.relu(self.stem1(obs))
        s = F.relu(self.n0(self.stem2(s)))
        x = self.c0(s)
        x = x + self.c1(F.relu(self.n1(x)))
        x = F.relu(self.nf(x))
        r = F.relu(self.skip(obs))
        f = torch.cat([x, r], dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(f, self.type_bias(g))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
