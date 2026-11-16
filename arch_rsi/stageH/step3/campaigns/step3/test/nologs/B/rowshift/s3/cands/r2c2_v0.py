import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.02, "lr": 0.0007, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16}

STEM = 64
W = 48
NBLK = 2
PROJ = 32
HID = 32
GLOB = 64


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(4, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(4, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * proj, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.normal_(o.weight, std=0.01)
            nn.init.zeros_(o.bias)

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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.npl = int(spec.n_place)
        cin = spec.in_channels + self.nt
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.Sequential(*[Block(W) for _ in range(NBLK)])
        self.norm = nn.GroupNorm(4, W)
        self.head = PoolHead(W, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * W, GLOB), nn.ReLU(),
                                  nn.Linear(GLOB, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value = nn.Sequential(nn.Linear(2 * W, GLOB), nn.ReLU(), nn.Linear(GLOB, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.npl].float().reshape(B, self.nt, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        f = F.relu(self.norm(self.blocks(s)))
        place = self.head(f)
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
