import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "n_envs": 512, "rollout_len": 16, "n_minibatches": 32, "lr": 0.0007}

STEM = 64
WIDTH = 40
PROJ = 32
HID = 32
NBLOCKS = 2
FLATC = 16


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(4, c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


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
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1))
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(NBLOCKS)])
        self.norm = nn.GroupNorm(4, WIDTH)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.flat_proj = nn.Conv2d(WIDTH, FLATC, 1)
        self.flat_out = nn.Linear(FLATC * self.Y * self.X + self.n_place, self.n_place)
        nn.init.normal_(self.flat_out.weight, std=0.01)
        nn.init.zeros_(self.flat_out.bias)
        self.done_head = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_head = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.normal_(self.done_head[2].weight, std=0.01)
        nn.init.zeros_(self.done_head[2].bias)

    def forward(self, obs, mask):
        m = mask[:, :self.n_place].to(obs.dtype)
        x = self.stem(obs)
        for blk in self.blocks:
            x = blk(x)
        f = F.relu(self.norm(x))
        fl = F.relu(self.flat_proj(f)).flatten(1)
        place = self.head(f) + self.flat_out(torch.cat([fl, m], dim=1))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
