import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95, "lr": 0.0015, "ent_coef": 0.003, "n_minibatches": 64}

E = 48
P = 16
H = 24
T2 = 32
G = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = int(spec.in_channels)
        self.register_buffer("zcode", torch.eye(self.Z))
        self.e1 = nn.Conv2d(12 + C + self.Z, E, 1)
        self.e2 = nn.Conv2d(E, E, 1)
        self.t0 = nn.Conv2d(C, T2, 1)
        self.t1 = nn.Conv2d(T2, T2, 3, padding=1)
        self.t2 = nn.Conv2d(T2, T2, 3, padding=1)
        self.tz = nn.Conv2d(T2, E, 1)
        self.proj = nn.Conv2d(E, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.orthogonal_(o.weight, 0.01)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * E + 2 * T2, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        lay = obs[:, :4 * Z].reshape(B, 4, Z, Y, X)
        below = F.pad(lay, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        above = F.pad(lay, (0, 0, 0, 0, 0, 1))[:, :, 1:]
        al = torch.cat([lay, below, above], 1).permute(0, 2, 1, 3, 4)
        col = obs.unsqueeze(1).expand(B, Z, C, Y, X)
        zc = self.zcode.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        inp = torch.cat([al, col, zc], 2).reshape(B * Z, 12 + C + Z, Y, X)
        t = F.relu(self.t0(obs))
        t = F.relu(self.t1(t))
        t = F.relu(t + self.t2(t))
        u = F.relu(self.e1(inp))
        tb = self.tz(t).unsqueeze(1).expand(B, Z, E, Y, X).reshape(B * Z, E, Y, X)
        u = F.relu(self.e2(u) + tb)
        p = self.proj(u)
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
        s = torch.cat(res, dim=1)
        nT = s.shape[1]
        s = s.reshape(B, Z, nT, Y, X)
        place = torch.logsumexp(s, dim=1).flatten(1)
        uu = u.reshape(B, Z, E, Y, X)
        g = self.glob(torch.cat([uu.mean((1, 3, 4)), uu.amax((1, 3, 4)), t.mean((2, 3)), t.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
