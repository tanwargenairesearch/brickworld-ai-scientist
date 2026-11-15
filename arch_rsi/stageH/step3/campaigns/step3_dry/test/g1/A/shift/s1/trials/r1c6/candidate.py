import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "n_minibatches": 16}

W = 16
P = 16
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = spec.Z
        self.Y = spec.Y
        self.X = spec.X
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.register_buffer("zpos", torch.eye(spec.Z).reshape(1, spec.Z, spec.Z, 1, 1))
        cin = 8 + 3 + spec.Z
        self.stem = nn.Conv2d(cin, W, 1)
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(3 * W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        for o in self.outs:
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done = nn.Linear(64, 1)
        nn.init.zeros_(self.done.weight)
        nn.init.zeros_(self.done.bias)
        self.value = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        L = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        below = torch.cat([torch.zeros_like(L[:, :1]), L[:, :-1]], dim=1)
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = self.zpos.expand(B, Z, Z, Y, X)
        inp = torch.cat([L, below, g3, zp], dim=2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.stem(inp))
        h = F.relu(h + self.c1(h))
        h5 = h.reshape(B, Z, W, Y, X)
        zero = torch.zeros_like(h5[:, :1])
        hb = torch.cat([zero, h5[:, :-1]], dim=1)
        ha = torch.cat([h5[:, 1:], zero], dim=1)
        hz = torch.cat([h5, hb, ha], dim=2).reshape(B * Z, 3 * W, Y, X)
        f = F.relu(h + self.c2(hz))
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
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1))))
            o = o.reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        f5 = f.reshape(B, Z, W, Y, X)
        g = self.glob(torch.cat([f5.mean((1, 3, 4)), f5.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
