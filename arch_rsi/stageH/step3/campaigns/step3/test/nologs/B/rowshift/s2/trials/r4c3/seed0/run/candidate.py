import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

WIDTH = 32
STEM = 64
NBLOCKS = 2
PROJ = 24
HID = 24
GHID = 64


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
            nn.init.orthogonal_(o.weight, 0.01)
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
        self.T = spec.n_types
        self.n_place = spec.n_place
        cin = spec.in_channels + spec.n_types
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1))
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(NBLOCKS)])
        self.out_norm = nn.GroupNorm(4, WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GHID), nn.ReLU())
        self.done_head = nn.Linear(GHID, 1)
        self.vh = nn.Sequential(nn.Linear(GHID, GHID), nn.ReLU())
        self.value_head = nn.Linear(GHID, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        x = self.stem(torch.cat([obs, m], dim=1))
        for b in self.blocks:
            x = b(x)
        f = F.relu(self.out_norm(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g)
        value = self.value_head(self.vh(g)).squeeze(1)
        return torch.cat([place, done], dim=1), value


def build_policy(spec):
    return Net(spec)
