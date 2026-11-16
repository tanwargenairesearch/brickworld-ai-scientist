import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 3, "ent_coef": 0.003, "gamma": 0.98, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
K = 6
PROJ = 16
HID = 32


def fp_pool(t, dy, dx):
    if dy * dx == 1:
        return t, t, t
    q = F.pad(t, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class LayerGateHead(nn.Module):
    def __init__(self, c, footprints, z):
        super().__init__()
        self.z = int(z)
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.lay = nn.Conv2d(c, K * self.z, 1)
        self.occ = nn.Conv2d(c, self.z, 1)
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.land = nn.ModuleList([nn.Conv2d(2 * self.z, self.z, 1) for _ in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(2 * K + 3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])

    def forward(self, f):
        b, _, yy, xx = f.shape
        z = self.z
        lay = self.lay(f)
        occ = self.occ(f)
        p = self.proj(f)
        res = []
        for (dy, dx), land, mix, out in zip(self.fps, self.land, self.mix, self.outs):
            lmx, lmn, _ = fp_pool(lay, dy, dx)
            omx, omn, _ = fp_pool(occ, dy, dx)
            pmx, pmn, pav = fp_pool(p, dy, dx)
            w = torch.softmax(land(torch.cat([omx, omn], dim=1)), dim=1).unsqueeze(1)
            gmx = (lmx.reshape(b, K, z, yy, xx) * w).sum(2)
            gmn = (lmn.reshape(b, K, z, yy, xx) * w).sum(2)
            h = F.relu(mix(torch.cat([gmx, gmn, pmx, pmn, pav], dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = LayerGateHead(WIDTH, spec.footprints, spec.Z)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
