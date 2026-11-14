import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
KEY = 24
HID = 32


class GatherHead(nn.Module):
    def __init__(self, c, footprints, key, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.cell = nn.Conv2d(c, hid, 1)
        self.key = nn.Conv2d(c, key, 1)
        self.agg = nn.ModuleList([nn.Conv2d(2 * key, hid, 1) for _ in self.fps])
        self.mix = nn.ModuleList([nn.Conv2d(3 * hid, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, f):
        Y = f.shape[2]
        X = f.shape[3]
        a = self.cell(f)
        k = self.key(f)
        res = []
        for (dy, dx), agg, mix, out in zip(self.fps, self.agg, self.mix, self.outs):
            kp = F.pad(k, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(kp, (dy, dx), stride=1)
            mn = -F.max_pool2d(-kp, (dy, dx), stride=1)
            u = agg(torch.cat([mx, mn], dim=1))
            ap = F.pad(a, (0, dx - 1, 0, dy - 1))
            hs = torch.stack([F.relu(ap[:, :, i:i + Y, j:j + X] + u)
                              for i in range(dy) for j in range(dx)], dim=0)
            h = torch.cat([hs.amax(0), hs.amin(0), hs.mean(0)], dim=1)
            h = F.relu(mix(h))
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
        self.head = GatherHead(WIDTH, spec.footprints, KEY, HID)
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
