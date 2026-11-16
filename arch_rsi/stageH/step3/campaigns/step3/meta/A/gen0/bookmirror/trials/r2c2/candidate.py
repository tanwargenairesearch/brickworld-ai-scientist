import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "gamma": 0.99, "lr": 0.002}

STEM = 64
COL = 48
TW = 32
K = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = int(spec.in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(C, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, TW, 3, padding=1)
        self.c2 = nn.Conv2d(TW, TW, 3, padding=1)
        self.lev = nn.Conv2d(C + COL + TW, self.Z * K, 1)
        self.lc = nn.Conv2d(K, K, 3, padding=1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * K, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TW + COL, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        t = F.relu(self.c1(s))
        t = F.relu(t + self.c2(t))
        h = F.relu(self.lev(torch.cat([obs, s, t], dim=1)))
        h = h.reshape(B * Z, K, Y, X)
        h = F.relu(h + self.lc(h))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = h, h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            o = o.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(o, dim=1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([t.mean((2, 3)), t.amax((2, 3)), s.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
