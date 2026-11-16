import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
AP = 16
AH = 16


def small_init(m):
    nn.init.orthogonal_(m.weight, gain=0.01)
    nn.init.zeros_(m.bias)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


def tf(x, k):
    if k & 4:
        x = x.transpose(2, 3)
    if k & 1:
        x = torch.flip(x, dims=[2])
    if k & 2:
        x = torch.flip(x, dims=[3])
    return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_t = len(spec.footprints)
        square = int(spec.Y) == int(spec.X)
        self.tlist = [1, 2, 3, 4, 5, 6, 7] if square else [1, 2, 3]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.aproj = nn.Conv2d(WIDTH, AP, 1)
        self.aconv = nn.ModuleList([nn.Conv2d(AP, AH, 3, padding=1) for _ in self.tlist])
        self.aout = nn.ModuleList([nn.Conv2d(AH, n_t, 1) for _ in self.tlist])
        for m in self.aout:
            small_init(m)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        small_init(self.done_head)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        a = F.relu(self.aproj(f))
        for k, conv, out in zip(self.tlist, self.aconv, self.aout):
            place = place + out(F.relu(conv(tf(a, k)))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        done = d + place.detach().amax(dim=1, keepdim=True)
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
