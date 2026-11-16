import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 64, "rollout_len": 8, "n_minibatches": 8, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.001}

STEM_H = 64
W = 32
PROJ = 32
HID = 32


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c)
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
        self.stem1 = nn.Conv2d(spec.in_channels, STEM_H, 1)
        self.stem2 = nn.Conv2d(STEM_H, W, 1)
        self.b1 = Block(W)
        self.b2 = Block(W)
        self.norm = nn.GroupNorm(8, W)
        self.head = PoolHead(2 * W, spec.footprints, PROJ, HID)
        self.done_mlp = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_out = nn.Linear(64, 1)
        nn.init.normal_(self.done_out.weight, std=0.01)
        nn.init.zeros_(self.done_out.bias)
        self.val_mlp = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.val_out = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem2(F.relu(self.stem1(obs)))
        f = self.b2(self.b1(s))
        f = F.relu(self.norm(f))
        place = self.head(torch.cat([F.relu(s), f], dim=1))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        done = self.done_out(self.done_mlp(g))
        value = self.val_out(self.val_mlp(g)).squeeze(1)
        logits = torch.cat([place, done], dim=1)
        return logits, value


def build_policy(spec):
    return Net(spec)
