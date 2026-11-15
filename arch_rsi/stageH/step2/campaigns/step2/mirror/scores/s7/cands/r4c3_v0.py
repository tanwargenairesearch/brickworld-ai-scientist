import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 16
HID = 24
PPROJ = 24
PHID = 24


class WindowHead(nn.Module):
    def __init__(self, c, n_types):
        super().__init__()
        self.proj = nn.Conv2d(c, PROJ, 1)
        self.conv = nn.Conv2d(2 * PROJ, n_types * HID, 3, padding=1)
        self.out = nn.Conv2d(n_types * HID, n_types, 1, groups=n_types)

    def forward(self, f):
        p = F.relu(self.proj(f))
        if p.shape[2] == p.shape[3]:
            pt = p.transpose(2, 3)
        else:
            pt = torch.zeros_like(p)
        q = torch.cat([p, pt], dim=1)
        h = F.relu(self.conv(q))
        return self.out(h).flatten(1)


class PoolHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, PPROJ, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * PPROJ, PHID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(PHID, 1, 1) for _ in self.fps])

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
        self.T = int(spec.n_types)
        self.NP = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + self.T, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.whead = WindowHead(WIDTH, self.T)
        self.phead = PoolHead(WIDTH, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.NP].to(obs.dtype).reshape(B, self.T, Y, X)
        inp = torch.cat([obs, m], dim=1)
        s = self.stem(inp)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.whead(f) + self.phead(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
