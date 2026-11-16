import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 7e-4, 'ppo_epochs': 4, 'gamma': 0.95, 'gae_lambda': 0.9, 'ent_coef': 0.01}

STEM = 64
D = 48
PROJ = 32
HID = 32
BLOCKS = 2


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


class MixBlock(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.t1 = nn.Linear(n, 2 * n)
        self.t2 = nn.Linear(2 * n, n)
        self.ln2 = nn.LayerNorm(d)
        self.c1 = nn.Linear(d, 2 * d)
        self.c2 = nn.Linear(2 * d, d)

    def forward(self, t):
        h = self.ln1(t).transpose(1, 2)
        h = self.t2(F.gelu(self.t1(h))).transpose(1, 2)
        t = t + h
        t = t + self.c2(F.gelu(self.c1(self.ln2(t))))
        return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec.Y * spec.X
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1))
        self.blocks = nn.ModuleList([MixBlock(n, D) for _ in range(BLOCKS)])
        self.lnf = nn.LayerNorm(D)
        self.head = PoolHead(D, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        s = self.stem(obs)
        t = s.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = blk(t)
        t = F.relu(self.lnf(t))
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
