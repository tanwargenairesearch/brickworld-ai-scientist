import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "gamma": 0.9, "ent_coef": 0.02}

MIXC = 24
STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
DONE_GAIN = 0.2


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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        Z = spec.Z
        self.Z = Z
        n_t = 3 * Z + 1
        self.pre = nn.Conv2d(n_t, MIXC, 1)
        self.mx = nn.Linear(spec.X, spec.X, bias=False)
        self.my = nn.Linear(spec.Y, spec.Y, bias=False)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + MIXC, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        nn.init.zeros_(self.done_head.weight)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        tgt = torch.cat([obs[:, Z:4 * Z], obs[:, 4 * Z + 1:4 * Z + 2]], dim=1)
        p = self.pre(tgt)
        p = self.mx(p)
        p = self.my(p.transpose(2, 3)).transpose(2, 3)
        a = F.relu(p)
        s = self.stem(torch.cat([obs, a], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        place = place - place.mean(1, keepdim=True)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = DONE_GAIN * self.done_head(g)
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
