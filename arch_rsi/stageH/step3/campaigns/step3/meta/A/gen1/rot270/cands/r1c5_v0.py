import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 64, "rollout_len": 64, "n_minibatches": 32, "lr": 0.002, "clip_eps": 0.3, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.005}

COLW = 64
WIDTH = 40
PROJ = 24
HID = 40
HID2 = 24
GL = 96
G = math.sqrt(2.0)


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        C = spec.in_channels + 2
        self.gu = init_(nn.Conv2d(C, COLW, 1), G)
        self.gv = init_(nn.Conv2d(C, COLW, 1), 1.0)
        self.c0 = init_(nn.Conv2d(COLW, COLW, 1), G)
        self.n0 = nn.GroupNorm(4, COLW)
        self.inp = init_(nn.Conv2d(COLW, WIDTH, 1), G)
        self.b1 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), G)
        self.bn1 = nn.GroupNorm(4, WIDTH)
        self.film = init_(nn.Linear(2 * WIDTH, WIDTH), 0.1)
        self.b2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), G)
        self.bn2 = nn.GroupNorm(4, WIDTH)
        self.proj = init_(nn.Conv2d(COLW + WIDTH, PROJ, 1), 1.0)
        self.fconv = nn.ModuleList([init_(nn.Conv2d(PROJ, HID, (dy, dx)), G) for dy, dx in self.fps])
        self.sconv = nn.ModuleList([init_(nn.Conv2d(2 * PROJ, HID, 1), G) for _ in self.fps])
        self.m2 = nn.ModuleList([init_(nn.Conv2d(HID, HID2, 1), G) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID2, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(COLW + WIDTH, 1, 1), 0.01)
        self.glob1 = init_(nn.Linear(2 * WIDTH + 2 * COLW, GL), G)
        self.glob2 = init_(nn.Linear(GL, GL), G)
        self.type_bias = init_(nn.Linear(GL, T), 0.01)
        self.done_head = init_(nn.Linear(GL, 1), 0.01)
        self.value_head = init_(nn.Linear(GL, 1), 1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        z = torch.cat([obs, ys, xs], 1)
        c = F.relu(self.gu(z)) * torch.sigmoid(self.gv(z))
        c = F.relu(self.n0(self.c0(c)))
        x = F.relu(self.inp(c))
        x = F.relu(x + self.bn1(self.b1(x)))
        gx = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        x = x + self.film(gx).view(B, WIDTH, 1, 1)
        f = F.relu(x + self.bn2(self.b2(x)))
        fr = torch.cat([c, f], 1)
        p = self.proj(fr)
        g = F.relu(self.glob1(torch.cat([f.mean((2, 3)), f.amax((2, 3)), c.mean((2, 3)), c.amax((2, 3))], 1)))
        g = F.relu(self.glob2(g))
        tb = self.type_bias(g)
        a = self.anchor(fr)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            if dy * dx == 1:
                mx, mn = p, p
            else:
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            h = F.relu(self.fconv[i](q) + self.sconv[i](torch.cat([mx, mn], 1)))
            h = F.relu(self.m2[i](h))
            res.append(self.outs[i](h) + a + tb[:, i].view(B, 1, 1, 1))
        place = torch.cat(res, 1).flatten(1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
