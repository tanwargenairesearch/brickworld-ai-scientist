import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

C = 24
P = 16
H = 16


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        Z = self.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.register_buffer("zpos", torch.eye(Z).reshape(1, Z, Z, 1, 1))
        self.inp = nn.Conv2d(7 + Z, C, 1)
        self.col1 = nn.Linear(2 * C, C)
        self.sp = nn.Conv2d(C, C, 3, padding=1)
        self.col2 = nn.Linear(2 * C, C)
        self.proj = nn.Conv2d(C, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * C, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def colmix(self, t, lin, B):
        Z = self.Z
        n, c, Y, X = t.shape
        u = t.reshape(B, Z, c, Y, X)
        s = torch.cat([u.mean(1), u.amax(1)], dim=1)
        s = lin(s.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        u = F.relu(u + s.unsqueeze(1))
        return u.reshape(B * Z, c, Y, X)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        layers = torch.stack([obs[:, k * Z:(k + 1) * Z] for k in range(4)], dim=2)
        shared = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zp = self.zpos.to(obs.dtype).expand(B, Z, Z, Y, X)
        tok = torch.cat([layers, shared, zp], dim=2).reshape(B * Z, 7 + Z, Y, X)
        t = F.relu(self.inp(tok))
        t = self.colmix(t, self.col1, B)
        t = F.relu(t + self.sp(t))
        t = self.colmix(t, self.col2, B)
        p = self.proj(t)
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
            o = out(h).reshape(B, Z, 2, Y, X)
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * o[:, :, 0]).sum(1).flatten(1))
        place = torch.cat(res, dim=1)
        u = t.reshape(B, Z, C, Y, X)
        g = self.glob(torch.cat([u.mean((1, 3, 4)), u.amax((1, 3, 4))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
