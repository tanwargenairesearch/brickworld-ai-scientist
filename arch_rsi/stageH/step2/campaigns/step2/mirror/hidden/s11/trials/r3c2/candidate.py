import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "n_minibatches": 32}

STEM = 64
COL = 48
WIDTH = 40
L = 16
LH = 16


class LayerHead(nn.Module):
    def __init__(self, Z, width, footprints, l, hid):
        super().__init__()
        self.Z = Z
        self.l = l
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.inp = nn.Conv2d(7 + Z, l, 1)
        self.ctx = nn.Conv2d(width, l, 1)
        self.dw = nn.Conv2d(l, l, 3, padding=1, groups=l)
        self.mix = nn.ModuleList([nn.Conv2d(2 * l, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, obs, f):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        per = torch.stack([b, t, m, e], dim=2)
        rest_e = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        u = torch.cat([per, rest_e, pos], dim=2).reshape(B * Z, 7 + Z, Y, X)
        c = self.ctx(f)
        h = self.inp(u).view(B, Z, self.l, Y, X) + c.unsqueeze(1)
        h = F.relu(h).reshape(B * Z, self.l, Y, X)
        h = F.relu(h + self.dw(h))
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn = h, h
            else:
                q = F.pad(h, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            o = self.outs[i](F.relu(self.mix[i](torch.cat([mx, mn], dim=1))))
            o = o.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(o, dim=1))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.lhead = LayerHead(spec.Z, WIDTH, spec.footprints, L, LH)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.lhead(obs, f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
