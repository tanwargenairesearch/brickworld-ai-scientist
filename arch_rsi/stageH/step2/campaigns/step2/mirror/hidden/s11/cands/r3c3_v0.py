import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "n_minibatches": 16, "gamma": 0.97, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
PROJ = 32
HID = 32
L = 24
LH = 24
GD = 64


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


class LayerHead(nn.Module):
    def __init__(self, Z, width, gdim, footprints, l, hid):
        super().__init__()
        self.Z = Z
        self.l = l
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.inp = nn.Conv2d(11 + Z, l, 1)
        self.ctx = nn.Conv2d(width, l, 1)
        self.gb = nn.Linear(gdim, l)
        self.dw = nn.Conv2d(l, l, 3, padding=1, groups=l)
        self.pw = nn.Conv2d(l, l, 1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * l, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, obs, f, g):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        rest = obs[:, 4 * Z:4 * Z + 3]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, 0:Z]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, 0:Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:Z + 1]
        ta = F.pad(t, (0, 0, 0, 0, 0, 1))[:, 1:Z + 1]
        per = torch.stack([b, t, m, e, bb, tb, ba, ta], dim=2)
        rest_e = rest.unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        u = torch.cat([per, rest_e, pos], dim=2).reshape(B * Z, 11 + Z, Y, X)
        c = self.ctx(f) + self.gb(g)[:, :, None, None]
        h = self.inp(u).view(B, Z, self.l, Y, X) + c.unsqueeze(1)
        h = F.relu(h).reshape(B * Z, self.l, Y, X)
        h = F.relu(h + self.dw(h))
        h = F.relu(h + self.pw(h))
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
        self.c3 = nn.Conv2d(WIDTH, WIDTH, 3, padding=2, dilation=2)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, GD), nn.ReLU())
        self.lhead = LayerHead(spec.Z, WIDTH, GD, spec.footprints, L, LH)
        self.done_head = nn.Linear(GD, 1)
        self.value_head = nn.Linear(GD, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        x = F.relu(x + self.c2(x))
        f = F.relu(x + self.c3(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        place = self.head(f) + self.lhead(obs, f, g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
