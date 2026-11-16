import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 7e-4, "ppo_epochs": 2, "n_minibatches": 16, "ent_coef": 0.01, "gamma": 0.97}

STEM = 64
W = 40
P = 32
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.X = int(spec.X)
        self.Y = int(spec.Y)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.ax = nn.Conv2d(W, W, (1, self.X))
        self.ay = nn.Conv2d(W, W, (self.Y, 1))
        self.c3 = nn.Conv2d(W, W, 3, padding=1, padding_mode="circular")
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(4 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        f = self.stem(obs)
        f = F.relu(f + self.c1(f))
        f = F.relu(f + self.c2(f))
        g = F.relu(self.ax(F.pad(f, (0, X - 1, 0, 0), mode="circular")))
        g = F.relu(self.ay(F.pad(g, (0, 0, 0, Y - 1), mode="circular")))
        g = F.relu(g + self.c3(g))
        p = self.proj(g)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1), mode="circular")
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        place = torch.cat(res, dim=1).reshape(B, -1)
        z = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), g.mean((2, 3)), g.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(z)], dim=1)
        return logits, self.value_head(z).squeeze(1)


def build_policy(spec):
    return Net(spec)
