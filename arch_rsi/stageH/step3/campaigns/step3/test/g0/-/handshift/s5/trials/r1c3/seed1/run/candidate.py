import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "ent_coef": 0.003, "gamma": 0.97}

COLW = 64
D = 32
P = 16
H = 24
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        self.col1 = nn.Conv2d(C, COLW, 1)
        self.col2 = nn.Conv2d(COLW, D, 1)
        self.lay = nn.Conv2d(13, D, 1)
        self.zemb = nn.Parameter(torch.randn(self.Z, D) * 0.02)
        self.l2 = nn.Conv2d(D, D, 1)
        self.proj = nn.Conv2d(D, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * COLW + 2 * D, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        lay5 = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(lay5, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(lay5, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        zc = torch.arange(Z, device=obs.device, dtype=obs.dtype) / max(Z - 1, 1)
        zc = zc.view(1, 1, Z, 1, 1).expand(B, 1, Z, Y, X)
        lin = torch.cat([below, lay5, above, zc], dim=1)
        lin = lin.permute(0, 2, 1, 3, 4).reshape(B * Z, 13, Y, X)

        c1 = F.relu(self.col1(obs))
        col = self.col2(c1)
        colrep = col.unsqueeze(1).expand(B, Z, D, Y, X).reshape(B * Z, D, Y, X)
        ze = self.zemb.view(1, Z, D, 1, 1).expand(B, Z, D, 1, 1).reshape(B * Z, D, 1, 1)
        h = F.relu(self.lay(lin) + colrep + ze)
        h = F.relu(h + self.l2(h))

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
            s = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            s = s.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.cat(res, dim=1)

        hz = h.reshape(B, Z * D, Y, X).reshape(B, Z, D, Y * X)
        gin = torch.cat([c1.mean((2, 3)), c1.amax((2, 3)),
                         hz.mean((1, 3)), hz.amax(3).amax(1)], dim=1)
        g = self.glob(gin)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
