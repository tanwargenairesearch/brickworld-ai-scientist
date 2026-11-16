import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "ppo_epochs": 4, "n_minibatches": 64, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.003}

STEM = 64
COL = 48
WIDTH = 40
RAW = 16
PROJ = 32
HID = 32
DCH = 8
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
        C = spec.in_channels + 2
        self.s1 = init_(nn.Conv2d(C, STEM, 1), G)
        self.s2 = init_(nn.Conv2d(STEM, COL, 1), G)
        self.c1 = init_(nn.Conv2d(COL, WIDTH, 3, padding=1), G)
        self.n1 = nn.GroupNorm(4, WIDTH)
        self.c2 = init_(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), G)
        self.n2 = nn.GroupNorm(4, WIDTH)
        self.raw = init_(nn.Conv2d(C, RAW, 1), G)
        self.proj = init_(nn.Conv2d(WIDTH + RAW, PROJ, 1), 1.0)
        self.mix = nn.ModuleList([init_(nn.Conv2d(3 * PROJ, HID, 1), G) for _ in self.fps])
        self.outs = nn.ModuleList([init_(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_(nn.Conv2d(WIDTH + RAW, 1, 1), 0.01)
        self.dproj = init_(nn.Conv2d(WIDTH + RAW, DCH, 1), G)
        self.dense = init_(nn.Linear(DCH * spec.Y * spec.X, spec.n_actions), 0.01)
        self.glob = init_(nn.Linear(2 * WIDTH, 64), G)
        self.type_bias = init_(nn.Linear(64, T), 0.01)
        self.done_head = init_(nn.Linear(64, 1), 0.01)
        self.value_head = init_(nn.Linear(64, 1), 1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        z = torch.cat([obs, ys, xs], 1)
        s = F.relu(self.s2(F.relu(self.s1(z))))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        fr = torch.cat([f, F.relu(self.raw(z))], 1)
        p = self.proj(fr)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)))
        tb = self.type_bias(g)
        a = self.anchor(fr)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = pool_stats(p, dy, dx)
            h = F.relu(self.mix[i](torch.cat([mx, mn, av], 1)))
            res.append(self.outs[i](h) + a + tb[:, i].view(B, 1, 1, 1))
        place = torch.cat(res, 1).flatten(1)
        structured = torch.cat([place, self.done_head(g)], 1)
        d = F.relu(self.dproj(fr)).flatten(1)
        logits = structured + self.dense(d)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
