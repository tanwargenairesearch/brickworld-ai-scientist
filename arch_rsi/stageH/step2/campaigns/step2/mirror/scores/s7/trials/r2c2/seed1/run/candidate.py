import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.003, "gamma": 0.97, "gae_lambda": 0.92}

STEM = 64
COL = 48
WIDTH = 32
LH = 48
LE = 32
PROJ = 12
HID = 24


class LayerHead(nn.Module):
    def __init__(self, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(LE, PROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PROJ, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])

    def forward(self, e):
        p = self.proj(e)
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
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.register_buffer("zpos", torch.eye(Z).view(1, Z, Z, 1, 1))
        self.l1 = nn.Conv2d(12 + Z, LH, 1)
        self.l2 = nn.Conv2d(LH, LE, 1)
        self.tproj = nn.Conv2d(WIDTH, LE, 1)
        self.head = LayerHead(spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        Z = self.Z
        build = obs[:, 0:Z]
        target = obs[:, Z:2 * Z]
        miss = obs[:, 2 * Z:3 * Z]
        extra = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        bb = F.pad(build, (0, 0, 0, 0, 1, 0))[:, :Z]
        tb = F.pad(target, (0, 0, 0, 0, 1, 0))[:, :Z]
        mb = F.pad(miss, (0, 0, 0, 0, 1, 0))[:, :Z]
        ba = F.pad(build, (0, 0, 0, 0, 0, 1))[:, 1:]
        ta = F.pad(target, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([build, target, miss, extra, bb, tb, mb, ba, ta], dim=2)
        rest_b = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = self.zpos.expand(B, Z, Z, Y, X)
        lin = torch.cat([per, rest_b, zp], dim=2).reshape(B * Z, 12 + Z, Y, X)

        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        t = self.tproj(f).unsqueeze(1).expand(B, Z, LE, Y, X).reshape(B * Z, LE, Y, X)
        e = F.relu(self.l2(F.relu(self.l1(lin))) + t)
        sc = self.head(e)
        T = sc.shape[1]
        sc = sc.reshape(B, Z, T, Y, X)
        place = torch.logsumexp(sc, dim=1).flatten(1)

        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
