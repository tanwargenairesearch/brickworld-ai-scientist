import torch
import torch.nn as nn
import torch.nn.functional as F

TR = 32
LW = 48
L = 32
PROJ = 16
HID = 16


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
        c = spec.in_channels
        self.lin = nn.Conv2d(c, LW, 1)
        self.zemb = nn.Embedding(self.Z, LW)
        self.lmix = nn.Conv2d(LW, L, 1)
        self.tin = nn.Conv2d(c, TR, 1)
        self.tc = nn.Conv2d(TR, TR, 3, padding=1)
        self.ctx = nn.Conv2d(TR, L, 1)
        self.head = PoolHead(L, spec.footprints, PROJ, HID)
        self.glob = nn.Sequential(nn.Linear(2 * TR + 2 * L, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        a = F.relu(self.tin(obs))
        f = F.relu(a + self.tc(a))
        u = self.lin(obs).unsqueeze(1) + self.zemb.weight.view(1, Z, LW, 1, 1)
        u = F.relu(u).reshape(B * Z, LW, Y, X)
        cx = self.ctx(f).unsqueeze(1).expand(B, Z, L, Y, X).reshape(B * Z, L, Y, X)
        t = F.relu(self.lmix(u) + cx)
        s = self.head(t).view(B, Z, -1)
        place = torch.logsumexp(s, dim=1)
        tv = t.view(B, Z, L, Y, X)
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                       tv.mean((1, 3, 4)), tv.amax(dim=(1, 3, 4))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
