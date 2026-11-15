import torch
import torch.nn as nn
import torch.nn.functional as F

STEM = 64
COL = 48
WIDTH = 40
LP = 24
LH = 16


class LayerHead(nn.Module):
    def __init__(self, spec, ctx_c, p, hid):
        super().__init__()
        self.Z = int(spec.Z)
        self.p = p
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.lin1 = nn.Linear(11, p)
        self.zemb = nn.Parameter(torch.randn(self.Z, p) * 0.1)
        self.ctx = nn.Conv2d(ctx_c, p, 1)
        self.lin2 = nn.Linear(3 * p, p)
        self.mix = nn.ModuleList([nn.Conv2d(3 * p, hid, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])

    def forward(self, obs, f):
        B, _, Y, X = obs.shape
        Z = self.Z
        p = self.p
        per = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=-1)
        below = F.pad(per, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        g = obs[:, 4 * Z:4 * Z + 3].permute(0, 2, 3, 1).unsqueeze(1).expand(B, Z, Y, X, 3)
        inp = torch.cat([per, below, g], dim=-1)
        c = self.ctx(f).permute(0, 2, 3, 1).unsqueeze(1)
        v = F.relu(self.lin1(inp) + self.zemb.view(1, Z, 1, 1, p) + c)
        cs = torch.cumsum(v, dim=1)
        below_agg = (cs - v) / Z
        above_agg = (cs[:, -1:] - cs) / Z
        u = F.relu(self.lin2(torch.cat([v, below_agg, above_agg], dim=-1)))
        u = u.permute(0, 1, 4, 2, 3).reshape(B * Z, p, Y, X)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = u, u, u
            else:
                q = F.pad(u, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            o = out(h).reshape(B, Z, Y, X)
            res.append(torch.logsumexp(o, dim=1).reshape(B, Y * X))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.head = LayerHead(spec, COL + WIDTH, LP, LH)
        self.glob = nn.Sequential(nn.Linear(2 * (COL + WIDTH), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        h = torch.cat([s, f], dim=1)
        place = self.head(obs, h)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
