import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 4, "lr": 0.001, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 24
SP = 16
SH = 24


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
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.ng = 8 if self.Y == self.X else 4
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.sproj = nn.Conv2d(WIDTH, SP, 1)
        self.sc1 = nn.Conv2d(SP, SH, 3, padding=1)
        self.sc2 = nn.Conv2d(SH, self.nt, 1)
        nn.init.zeros_(self.sc2.weight)
        nn.init.zeros_(self.sc2.bias)
        self.gate = nn.Parameter(torch.zeros(2 * self.ng))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        p = F.relu(self.sproj(f))
        w = torch.softmax(self.gate, dim=0) * 4.0
        extra = torch.zeros_like(place)
        for g in range(self.ng):
            q = p
            if g >= 4:
                q = q.transpose(2, 3)
            if g % 2 == 1:
                q = q.flip(2)
            if (g // 2) % 2 == 1:
                q = q.flip(3)
            sc = self.sc2(F.relu(self.sc1(q)))
            a = sc.flatten(1)
            b = sc.permute(0, 2, 3, 1).flatten(1)
            extra = extra + w[2 * g] * a + w[2 * g + 1] * b
        place = place + extra
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
