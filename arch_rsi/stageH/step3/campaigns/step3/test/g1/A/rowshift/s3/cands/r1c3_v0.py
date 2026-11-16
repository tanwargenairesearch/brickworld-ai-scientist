import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.003, "n_envs": 128, "rollout_len": 64, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.003, "gamma": 0.995, "gae_lambda": 0.99, "clip_eps": 0.3}

A = 96
CF = 48
CTX = 32
PROJ = 32
HID = 64
HID2 = 32
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
        self.m1 = init_(nn.Conv2d(cin, A, 1), RG)
        self.g1 = norm(A)
        self.m2 = init_(nn.Conv2d(A, A, 1), RG)
        self.g2 = norm(A)
        self.m3 = init_(nn.Conv2d(A, CF, 1), RG)
        self.ctx = init_(nn.Conv2d(CF, CTX, 3, padding=1), RG)
        self.gc = norm(CTX)
        hc = CF + CTX
        self.proj = init_(nn.Conv2d(hc, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), RG) for _ in self.fps])
        self.mix2 = nn.ModuleList([init_(nn.Conv2d(HID, HID2, 1), RG) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID2, 1, 1), 0.01) for _ in self.fps])
        self.gnorm = nn.LayerNorm(2 * hc)
        self.glob = init_(nn.Linear(2 * hc, G), RG)
        self.type_head = init_(nn.Linear(G, nt), 0.01)
        self.done_head = init_(nn.Linear(G, 1), 0.01)
        self.value_head = init_(nn.Linear(G, 1), 1.0)

    def forward(self, obs, mask):
        a = F.relu(self.g1(self.m1(obs)))
        a = F.relu(self.g2(self.m2(a)))
        c = F.relu(self.m3(a))
        t = F.relu(self.gc(self.ctx(c)))
        h = torch.cat([c, t], dim=1)
        g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = F.relu(self.glob(self.gnorm(g)))
        tb = self.type_head(g)
        p = self.proj(h)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            z = torch.cat(fp_pool(p, dy, dx), dim=1)
            z = F.relu(self.mix[i](z))
            z = F.relu(self.mix2[i](z))
            res.append(self.outs[i](z) + tb[:, i].view(-1, 1, 1, 1))
        place = torch.cat(res, dim=1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
