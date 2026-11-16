import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
K = 8
P = 24
HID = 32

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.95, "clip_eps": 0.2}


def pools(t, dy, dx):
    if dy * dx == 1:
        return t, t, t
    q = F.pad(t, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class LandHead(nn.Module):
    def __init__(self, c, footprints, Z):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.Z = Z
        self.hproj = nn.Conv2d(c, Z, 1)
        self.vproj = nn.Conv2d(c, Z * K, 1)
        self.gproj = nn.Conv2d(c, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * K + 3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])

    def forward(self, f):
        B, _, Y, X = f.shape
        Z = self.Z
        h = torch.sigmoid(self.hproj(f))
        v = self.vproj(f)
        g = self.gproj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            hp, _, _ = pools(h, dy, dx)
            nxt = torch.cat([hp[:, 1:], torch.zeros_like(hp[:, :1])], dim=1)
            p = F.relu(hp - nxt)
            vx, vn, va = pools(v, dy, dx)
            vm = torch.stack([vx, vn, va], dim=1).reshape(B, 3, Z, K, Y, X)
            sel = (vm * p.reshape(B, 1, Z, 1, Y, X)).sum(2).reshape(B, 3 * K, Y, X)
            gx, gn, ga = pools(g, dy, dx)
            hh = F.relu(mix(torch.cat([sel, gx, gn, ga], dim=1)))
            res.append(out(hh))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = LandHead(WIDTH, spec.footprints, int(spec.Z))
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        nn.init.orthogonal_(self.done_head.weight, gain=0.01)
        nn.init.zeros_(self.done_head.bias)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        d = self.done_head(g)
        lp_place = F.log_softmax(place, dim=1) + F.logsigmoid(-d)
        lp_done = F.logsigmoid(d)
        logits = torch.cat([lp_place, lp_done], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
