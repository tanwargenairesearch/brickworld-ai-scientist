import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 7e-4, "ppo_epochs": 2, "n_minibatches": 16, "ent_coef": 0.01, "gamma": 0.97}

STEM = 64
W = 48
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
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.sx = nn.Conv2d(HID, HID, (1, self.X), groups=HID)
        self.sy = nn.Conv2d(HID, HID, (self.Y, 1), groups=HID)
        self.out1 = nn.Conv2d(HID, HID, 1)
        self.out2 = nn.Conv2d(HID, 1, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        f = self.stem(obs)
        f = F.relu(f + self.c1(f))
        f = F.relu(f + self.c2(f))
        p = self.proj(f)
        hs = []
        for (dy, dx), mix in zip(self.fps, self.mix):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            hs.append(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
        T = len(hs)
        h = torch.stack(hs, 1).reshape(B * T, HID, Y, X)
        h = self.sx(F.pad(h, (0, X - 1, 0, 0), mode="circular"))
        h = self.sy(F.pad(h, (0, 0, 0, Y - 1), mode="circular"))
        o = self.out2(F.relu(self.out1(h)))
        place = o.reshape(B, T * Y * X)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
