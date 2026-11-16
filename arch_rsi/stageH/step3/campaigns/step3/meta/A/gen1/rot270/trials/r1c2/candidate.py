import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.003, "clip_eps": 0.3, "ppo_epochs": 3, "n_minibatches": 64, "ent_coef": 0.005}

STEM = 64
COL = 48
WIDTH = 40
RAW = 16
PROJ = 32
HID = 48
HID2 = 32
G = 1.4142135623730951


def init_(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def pool_stats(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        C = spec.in_channels
        self.s1 = init_(nn.Conv2d(C, STEM, 1), G)
        self.s2 = init_(nn.Conv2d(STEM, COL, 1), G)
        self.n0 = nn.GroupNorm(4, COL)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), G)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), G)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.raw = init_(nn.Conv2d(C, RAW, 1), G)
        self.proj = init_(nn.Conv2d(WIDTH + RAW, PROJ, 1), 1.0)
        self.m1 = init_(nn.Conv2d(3 * PROJ, HID, 1), G)
        self.temb = nn.Parameter(torch.zeros(T, HID))
        self.m2 = init_(nn.Conv2d(HID, HID2, 1), G)
        self.shared_out = init_(nn.Conv2d(HID2, 1, 1), 0.01)
        self.type_out = nn.ModuleList([init_(nn.Conv2d(HID2, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(WIDTH + RAW, 1, 1), 0.01)
        self.glob = init_(nn.Linear(2 * WIDTH, 64), G)
        self.type_bias = init_(nn.Linear(64, T), 0.01)
        self.done_head = init_(nn.Linear(64, 1), 0.01)
        self.value_head = init_(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T = len(self.fps)
        s = F.relu(self.n0(self.s2(F.relu(self.s1(obs)))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        fr = torch.cat([f, F.relu(self.raw(obs))], 1)
        p = self.proj(fr)
        Y, X = p.shape[2], p.shape[3]
        stats = []
        for dy, dx in self.fps:
            mx, mn, av = pool_stats(p, dy, dx)
            stats.append(torch.cat([mx, mn, av], 1))
        S = torch.stack(stats, 1).reshape(B * T, 3 * PROJ, Y, X)
        h = self.m1(S).view(B, T, HID, Y, X) + self.temb.view(1, T, HID, 1, 1)
        h = F.relu(h).reshape(B * T, HID, Y, X)
        h2 = F.relu(self.m2(h))
        sh = self.shared_out(h2).view(B, T, Y, X)
        h2 = h2.view(B, T, HID2, Y, X)
        to = torch.cat([self.type_out[i](h2[:, i]) for i in range(T)], 1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        tb = self.type_bias(g).view(B, T, 1, 1)
        a = self.anchor(fr)
        place = (sh + to + a + tb).flatten(1)
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
