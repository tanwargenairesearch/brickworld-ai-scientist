import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.003, "n_envs": 128, "rollout_len": 8, "n_minibatches": 4, "ppo_epochs": 1, "clip_eps": 0.3, "ent_coef": 0.005, "gamma": 0.95, "gae_lambda": 0.9}

H1 = 64
H2 = 48
H3 = 24
PROJ = 32
HID = 32


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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, H1, 1), nn.ReLU(),
            nn.Conv2d(H1, H2, 1), nn.ReLU())
        self.c1 = nn.Conv2d(H2, H3, 3, padding=1)
        self.head = PoolHead(H2 + H3, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * (H2 + H3), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        for o in self.head.outs:
            nn.init.orthogonal_(o.weight, 0.01)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.constant_(self.done_head.bias, math.log(max(int(spec.n_place), 2)) - 2.0)
        nn.init.orthogonal_(self.value_head.weight, 1.0)

    def forward(self, obs, mask):
        s = self.stem(obs)
        f = torch.cat([s, F.relu(self.c1(s))], dim=1)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
