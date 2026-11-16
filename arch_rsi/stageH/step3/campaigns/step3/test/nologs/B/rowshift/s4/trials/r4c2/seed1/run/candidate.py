import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
GH = 64

RECIPE = {"ppo_epochs": 1, "gamma": 0.98, "gae_lambda": 0.95, "ent_coef": 0.01, "n_envs": 512, "rollout_len": 8, "n_minibatches": 32}


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
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.done = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU(), nn.Linear(GH, 1))
        nn.init.zeros_(self.done[2].weight)
        nn.init.zeros_(self.done[2].bias)
        self.value = nn.Sequential(nn.Linear(2 * WIDTH, GH), nn.ReLU(), nn.Linear(GH, 1))

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        d = self.done(g)
        lp = F.log_softmax(place, dim=1) + F.logsigmoid(-d)
        logits = torch.cat([lp, F.logsigmoid(d)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
