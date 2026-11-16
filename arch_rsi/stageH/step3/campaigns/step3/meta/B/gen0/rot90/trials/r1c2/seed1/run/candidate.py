import torch
import torch.nn as nn
import torch.nn.functional as F

TR = 32
D = 24
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.trunk_in = nn.Conv2d(spec.in_channels, TR, 1)
        self.t1 = nn.Conv2d(TR, TR, 3, padding=1)
        self.t2 = nn.Conv2d(TR, TR, 3, padding=1)
        self.ctx = nn.Conv2d(TR, D, 1)
        self.enc1 = nn.Conv2d(15 + Z, D, 1)
        self.enc2 = nn.Conv2d(D, D, 1)
        self.mix = nn.ModuleList([nn.Conv2d(2 * D, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * TR + 2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        self.register_buffer('zpos', torch.eye(Z))

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        Z = self.Z
        t = F.relu(self.trunk_in(obs))
        t = F.relu(t + self.t1(t))
        t = F.relu(t + self.t2(t))
        layers = torch.stack([obs[:, 0:Z], obs[:, Z:2 * Z], obs[:, 2 * Z:3 * Z], obs[:, 3 * Z:4 * Z]], dim=2)
        below = F.pad(layers, (0, 0, 0, 0, 0, 0, 1, 0))[:, :Z]
        above = F.pad(layers, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:]
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = self.zpos.view(1, Z, Z, 1, 1).expand(B, Z, Z, Y, X)
        raw = torch.cat([layers, below, above, gl, zp], dim=2).reshape(B * Z, 15 + Z, Y, X)
        ctx = self.ctx(t).unsqueeze(1).expand(B, Z, D, Y, X).reshape(B * Z, D, Y, X)
        u = F.relu(self.enc1(raw) + ctx)
        u = F.relu(u + self.enc2(u))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn = u, u
            else:
                q = F.pad(u, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            s = out(F.relu(mix(torch.cat([mx, mn], dim=1))))
            s = s.reshape(B, Z, Y * X)
            res.append(torch.logsumexp(s, dim=1))
        place = torch.cat(res, dim=1)
        ul = u.reshape(B, Z, D, Y, X)
        g = torch.cat([t.mean((2, 3)), t.amax((2, 3)), ul.mean((1, 3, 4)), ul.amax((1, 3, 4))], dim=1)
        g = self.glob(g)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
