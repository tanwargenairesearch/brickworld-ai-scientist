import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
LH = 32

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 3, "lr": 0.001, "gamma": 0.97}


class LayerStem(nn.Module):
    def __init__(self, Z, hid):
        super().__init__()
        self.Z = Z
        self.inp = nn.Conv2d(8, hid, 1)
        self.glob_in = nn.Conv2d(3, hid, 1)
        self.zemb = nn.Parameter(torch.zeros(1, Z, hid, 1, 1))
        self.mid = nn.Conv2d(hid, hid, 1)

    def forward(self, obs):
        Z = self.Z
        B, C, Y, X = obs.shape
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bp = F.pad(b, (0, 0, 0, 0, 1, 1))
        tp = F.pad(t, (0, 0, 0, 0, 1, 1))
        per = torch.stack([b, bp[:, 0:Z], bp[:, 2:Z + 2], t, tp[:, 0:Z], tp[:, 2:Z + 2], m, e], dim=2)
        per = per.reshape(B * Z, 8, Y, X)
        h = self.inp(per).reshape(B, Z, -1, Y, X)
        g = self.glob_in(obs[:, 4 * Z:4 * Z + 3]).unsqueeze(1)
        h = F.relu(h + g + self.zemb)
        h = F.relu(self.mid(h.reshape(B * Z, -1, Y, X))).reshape(B, Z, -1, Y, X)
        return torch.cat([h.mean(1), h.amax(1)], dim=1)


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
        self.lstem = LayerStem(spec.Z, LH)
        self.dstem = nn.Sequential(nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU())
        self.fuse = nn.Conv2d(STEM + 2 * LH, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = F.relu(self.fuse(torch.cat([self.dstem(obs), self.lstem(obs)], dim=1)))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
