import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 3}

STEM = 64
COL = 48
WIDTH = 40
LW = 32
P = 16
HID = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.lin = nn.Conv2d(8, LW, 1)
        self.cold = nn.Conv2d(spec.in_channels, LW, 1)
        self.colp = nn.Conv2d(WIDTH, LW, 1)
        self.zemb = nn.Embedding(self.Z, LW)
        self.l2 = nn.Conv2d(LW, LW, 1)
        self.proj = nn.Conv2d(LW, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        ta = F.pad(t, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([b, bb, ba, t, tb, ta, m, e], dim=2).reshape(B * Z, 8, Y, X)
        h = self.lin(per).reshape(B, Z, LW, Y, X)
        zid = torch.arange(Z, device=obs.device)
        ctx = self.cold(obs) + self.colp(f)
        h = h + self.zemb(zid).view(1, Z, LW, 1, 1) + ctx.unsqueeze(1)
        h = F.relu(h).reshape(B * Z, LW, Y, X)
        h = F.relu(self.l2(h))
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
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            o = o.reshape(B, Z, 2, Y, X)
            sc = o[:, :, 0]
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * sc).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
