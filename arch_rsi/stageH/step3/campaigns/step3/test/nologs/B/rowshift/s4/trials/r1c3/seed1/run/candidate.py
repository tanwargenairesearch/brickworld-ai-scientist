import torch
import torch.nn as nn
import torch.nn.functional as F

D = 16
WIDTH = 40
PROJ = 24
HID = 24
GH = 64

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.003, "n_envs": 512, "rollout_len": 8, "n_minibatches": 32}


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


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(4, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = int(spec.Z)
        self.Z = Z
        self.n_extra = int(spec.in_channels) - 4 * Z
        self.l1 = nn.Conv2d(5, D, (3, 1), padding=(1, 0))
        self.l2 = nn.Conv2d(D, D, (3, 1), padding=(1, 0))
        zc = torch.arange(Z, dtype=torch.float32) / max(Z - 1, 1)
        self.register_buffer("zc", zc.view(1, 1, Z, 1))
        self.stem = nn.Conv2d(D * Z + self.n_extra, WIDTH, 1)
        self.blocks = nn.Sequential(Block(WIDTH), Block(WIDTH))
        self.norm = nn.GroupNorm(4, WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.done = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU(), nn.Linear(GH, 1))
        self.value = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU(), nn.Linear(GH, 1))
        nn.init.orthogonal_(self.done[2].weight, 0.01)
        nn.init.zeros_(self.done[2].bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y * X)
        zc = self.zc.expand(B, 1, Z, Y * X)
        h = F.relu(self.l1(torch.cat([lay, zc], 1)))
        h = F.relu(self.l2(h))
        h = h.reshape(B, D * Z, Y, X)
        f = self.stem(torch.cat([h, obs[:, 4 * Z:]], 1))
        f = self.blocks(f)
        f = F.relu(self.norm(f))
        place = self.head(f)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
