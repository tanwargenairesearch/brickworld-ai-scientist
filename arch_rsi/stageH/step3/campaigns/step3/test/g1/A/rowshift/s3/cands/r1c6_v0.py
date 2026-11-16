import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.9, "clip_eps": 0.25}

STEM = 64
W = 40
LOW = 64
RAW = 16
PROJ = 24
HID = 32
G = 64
RG = math.sqrt(2.0)


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def norm(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


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
        cin = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        nt = len(self.fps)
        self.stem1 = init_(nn.Conv2d(cin, STEM, 1), RG)
        self.stem2 = init_(nn.Conv2d(STEM, W, 1), RG)
        self.c1 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n1 = norm(W)
        self.down = init_(nn.Conv2d(W, LOW, 3, stride=2, padding=1), RG)
        self.nd = norm(LOW)
        self.low = init_(nn.Conv2d(LOW, LOW, 3, padding=1), RG)
        self.nl = norm(LOW)
        self.up = init_(nn.Conv2d(LOW, W, 1), 1.0)
        self.c2 = init_(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = norm(W)
        self.raw = init_(nn.Conv2d(cin, RAW, 1), 1.0)
        hc = W + RAW
        self.proj = init_(nn.Conv2d(hc, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(hc, 1, 1), 0.01)
        gin = 2 * W + LOW + RAW
        self.gnorm = nn.LayerNorm(gin)
        self.glob = init_(nn.Linear(gin, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        s = F.relu(self.stem2(F.relu(self.stem1(obs))))
        x = F.relu(self.n1(self.c1(s)))
        d = F.relu(self.nd(self.down(x)))
        d = F.relu(d + self.nl(self.low(d)))
        u = F.interpolate(d, size=(Y, X), mode="nearest")
        x2 = F.relu(x + self.up(u))
        f = F.relu(x2 + self.n2(self.c2(x2)))
        r = self.raw(obs)
        hc = torch.cat([f, r], dim=1)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), d.amax((2, 3)), r.amax((2, 3))], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        anc = self.anchor(hc)
        p = self.proj(hc)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat(fp_pool(p, dy, dx), dim=1)
            z = F.relu(self.mix[i](z))
            res.append(self.outs[i](z) + anc + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
