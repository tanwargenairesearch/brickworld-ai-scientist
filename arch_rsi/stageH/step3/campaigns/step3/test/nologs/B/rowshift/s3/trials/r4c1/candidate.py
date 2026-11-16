import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

STEM = 64
W = 32
HID = 32
NBLK = 2
GLOB = 64
K = 8


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


def layouts(m):
    B = m.shape[0]
    res = []
    for tr in (False, True):
        base = m.transpose(2, 3) if tr else m
        for fy in (False, True):
            for fx in (False, True):
                v = base
                if fy:
                    v = torch.flip(v, [2])
                if fx:
                    v = torch.flip(v, [3])
                res.append(v.reshape(B, -1))
    return res


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1))
        self.blocks = nn.Sequential(*[Block(W) for _ in range(NBLK)])
        self.norm = nn.GroupNorm(4, W)
        self.glob = nn.Sequential(nn.Linear(2 * W, GLOB), nn.ReLU(),
                                  nn.Linear(GLOB, GLOB), nn.ReLU())
        self.gproj = nn.Linear(GLOB, HID)
        self.hconv = nn.Conv2d(W, HID, 3, padding=1)
        self.out = nn.Conv2d(HID, K * self.nt, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.done_head = nn.Linear(GLOB, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value = nn.Sequential(nn.Linear(2 * W, GLOB), nn.ReLU(), nn.Linear(GLOB, 1))

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        f = F.relu(self.norm(self.blocks(self.stem(obs))))
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        h = F.relu(self.hconv(f) + self.gproj(g)[:, :, None, None])
        m = self.out(h).reshape(B, K, self.nt, Y, X)
        place = None
        for k in range(K):
            v = layouts(m[:, k])[k]
            place = v if place is None else place + v
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
