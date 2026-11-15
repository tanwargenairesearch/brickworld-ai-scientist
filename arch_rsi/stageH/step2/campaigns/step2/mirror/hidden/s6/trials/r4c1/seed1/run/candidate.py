import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
H = 32
P = 16
HID = 16

RECIPE = {"n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 2, "lr": 0.001, "gae_lambda": 0.95, "gamma": 0.97}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.ctx = nn.Conv2d(WIDTH, H, 1)
        self.inp = nn.Conv2d(8, H, 1)
        self.zemb = nn.Parameter(torch.zeros(1, spec.Z, H, 1, 1))
        self.mid = nn.Conv2d(H, H, 1)
        self.proj = nn.Conv2d(H, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        Z = self.Z
        B, C, Y, X = obs.shape
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bdown = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)
        bup = F.pad(b, (0, 0, 0, 0, 0, 1))
        tp = F.pad(t, (0, 0, 0, 0, 1, 1))
        per = torch.stack([b, bdown[:, 0:Z], bup[:, 1:Z + 1], t, tp[:, 0:Z], tp[:, 2:Z + 2], m, e], dim=2)
        h = self.inp(per.reshape(B * Z, 8, Y, X)).reshape(B, Z, H, Y, X)
        h = F.relu(h + self.ctx(f).unsqueeze(1) + self.zemb)
        h = h.reshape(B * Z, H, Y, X)
        h = F.relu(h + self.mid(h))
        p = self.proj(h)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))).reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
