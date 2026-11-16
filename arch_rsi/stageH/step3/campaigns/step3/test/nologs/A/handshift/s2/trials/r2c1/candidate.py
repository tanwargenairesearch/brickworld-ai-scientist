import torch
import torch.nn as nn
import torch.nn.functional as F

W = 24
P = 16
H = 16
G = 64

RECIPE = {"ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.005}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        cl = 11 + self.Z
        self.inp = nn.Conv2d(cl, W, 1)
        self.ctx = nn.Conv2d(W, W, 1)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU(), nn.Linear(G, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]
        bb = F.pad(b, (0, 0, 0, 0, 1, 0))[:, :Z]
        ba = F.pad(b, (0, 0, 0, 0, 0, 1))[:, 1:]
        tb = F.pad(t, (0, 0, 0, 0, 1, 0))[:, :Z]
        mb = F.pad(m, (0, 0, 0, 0, 1, 0))[:, :Z]
        per = torch.stack([b, bb, ba, t, tb, m, mb, e], dim=2)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        eye = torch.eye(Z, device=obs.device, dtype=obs.dtype).view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        x = torch.cat([per, gl, eye], dim=2).reshape(B * Z, 11 + Z, Y, X)
        h = F.relu(self.inp(x))
        c = h.view(B, Z, W, Y, X).amax(1)
        c = self.ctx(c).unsqueeze(1).expand(B, Z, W, Y, X).reshape(B * Z, W, Y, X)
        h = h + F.relu(self.c3(h) + c)
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
            res.append(out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))))
        s = torch.cat(res, dim=1)
        T = s.shape[1]
        s = s.view(B, Z, T, Y, X).logsumexp(dim=1)
        place = s.flatten(1)
        hz = h.view(B, Z, W, Y, X)
        g = torch.cat([hz.mean((1, 3, 4)), hz.amax((1, 3, 4))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
