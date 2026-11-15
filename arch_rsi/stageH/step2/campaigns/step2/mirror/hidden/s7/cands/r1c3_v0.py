import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3}

LW = 32
DIRECT = 32
COL = 48
WIDTH = 40
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
        self.Z = int(spec.Z)
        Z = self.Z
        k = 8 + 3 + Z
        self.register_buffer("zpos", torch.eye(Z).view(1, Z, Z, 1, 1))
        self.layer = nn.Sequential(
            nn.Conv2d(k, LW, 1), nn.ReLU(),
            nn.Conv2d(LW, LW, 1), nn.ReLU())
        self.att = nn.Conv2d(LW, 1, 1)
        self.direct = nn.Conv2d(spec.in_channels, DIRECT, 1)
        self.merge = nn.Conv2d(3 * LW + DIRECT, COL, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = PoolHead(WIDTH, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ta = F.pad(t, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([b, bb, ba, t, tb, ta, m, e], dim=2)
        g = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = self.zpos.to(obs.dtype).expand(B, Z, Z, Y, X)
        h = torch.cat([per, g, zp], dim=2).reshape(B * Z, 11 + Z, Y, X)
        h = self.layer(h)
        a = self.att(h).reshape(B, Z, 1, Y, X)
        h = h.reshape(B, Z, LW, Y, X)
        w = torch.softmax(a, dim=1)
        hs = (w * h).sum(1)
        hm = h.amax(1)
        ha = h.mean(1)
        d = F.relu(self.direct(obs))
        s = F.relu(self.merge(torch.cat([hs, hm, ha, d], dim=1)))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        place = self.head(f)
        gg = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
