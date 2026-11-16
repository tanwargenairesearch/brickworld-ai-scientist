import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.0005, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.01}

STEM = 64
W = 40
PROJ = 32
HID = 32
VW = 32


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


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
        self.s1 = nn.Conv2d(spec.in_channels, STEM, 1)
        self.n1 = CLN(STEM)
        self.s2 = nn.Conv2d(STEM, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.n2 = CLN(W)
        self.head = PoolHead(W, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.v1 = nn.Conv2d(spec.in_channels, VW, 1)
        self.vn = CLN(VW)
        self.v2 = nn.Conv2d(VW, VW, 3, padding=1)
        self.vfc = nn.Sequential(nn.Linear(2 * VW, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = F.relu(self.n1(self.s1(obs)))
        s = F.relu(self.s2(s))
        x = F.relu(s + self.c1(s))
        f = F.relu(self.n2(x + self.c2(x)))
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        v = F.relu(self.vn(self.v1(obs)))
        v = F.relu(v + self.v2(v))
        vg = self.vfc(torch.cat([v.mean((2, 3)), v.amax((2, 3))], dim=1))
        return logits, self.value_head(vg).squeeze(1)


def build_policy(spec):
    return Net(spec)
