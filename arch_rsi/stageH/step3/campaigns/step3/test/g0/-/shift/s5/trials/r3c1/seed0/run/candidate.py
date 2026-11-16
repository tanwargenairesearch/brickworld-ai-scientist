import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 12
HID = 48


def window_bank(footprints):
    shapes = []
    for a, b in footprints:
        for s in ((int(a), int(b)), (int(b), int(a))):
            if s not in shapes:
                shapes.append(s)
    return shapes


def aligns(n):
    return [0] if n == 1 else [0, n - 1]


class BankHead(nn.Module):
    def __init__(self, c, spec, proj, hid):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.shapes = window_bank(spec.footprints)
        n_sets = 0
        for a, b in self.shapes:
            n_sets += len(aligns(a)) * len(aligns(b))
        self.proj = nn.Conv2d(c, proj, 1)
        self.mix = nn.Conv2d(3 * proj * n_sets + c, hid, 1)
        self.out = nn.Conv2d(hid, 2 * self.nt, 1)

    def forward(self, f):
        Y, X, T = self.Y, self.X, self.nt
        p = self.proj(f)
        feats = [f]
        for a, b in self.shapes:
            q = F.pad(p, (b - 1, b - 1, a - 1, a - 1))
            mx = F.max_pool2d(q, (a, b), stride=1)
            mn = -F.max_pool2d(-q, (a, b), stride=1)
            av = F.avg_pool2d(q, (a, b), stride=1)
            st = torch.cat([mx, mn, av], dim=1)
            for oy in aligns(a):
                for ox in aligns(b):
                    feats.append(st[:, :, oy:oy + Y, ox:ox + X])
        h = F.relu(self.mix(torch.cat(feats, dim=1)))
        o = self.out(h)
        direct = o[:, :T].flatten(1)
        trans = o[:, T:].transpose(2, 3).flatten(1)
        return direct + trans


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = BankHead(WIDTH, spec, PROJ, HID)
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
