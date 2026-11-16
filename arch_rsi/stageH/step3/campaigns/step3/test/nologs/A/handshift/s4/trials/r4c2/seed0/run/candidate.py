import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "gamma": 0.97, "gae_lambda": 0.95, "lr": 0.0003, "ent_coef": 0.001, "n_envs": 512, "rollout_len": 16, "n_minibatches": 32}

W = 20
NB = 2
P = 12
HID = 16
GH = 64
NIN = 14


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.inp = nn.Conv2d(NIN, W, 1)
        self.convs = nn.ModuleList([nn.Conv2d(W, W, 3, padding=1) for _ in range(NB)])
        self.ctxs = nn.ModuleList([nn.Linear(W, W) for _ in range(NB)])
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, HID, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 2, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * W + self.Z * W, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z, Y, X = self.Z, self.Y, self.X
        b = obs[:, 0:Z]
        t = obs[:, Z:2 * Z]
        m = obs[:, 2 * Z:3 * Z]
        e = obs[:, 3 * Z:4 * Z]

        def dn(a):
            return F.pad(a, (0, 0, 0, 0, 1, 0))[:, :Z]

        def up(a):
            return F.pad(a, (0, 0, 0, 0, 0, 1))[:, 1:]

        per = torch.stack([b, t, m, e, dn(b), dn(t), dn(m), up(b), up(t), up(m)], dim=2)
        sc = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        x = torch.cat([per, sc, zc], dim=2).reshape(B * Z, NIN, Y, X)
        h = F.relu(self.inp(x))
        for conv, ctx in zip(self.convs, self.ctxs):
            g = ctx(h.reshape(B, Z, W, Y, X).mean((1, 3, 4)))
            g = g.unsqueeze(1).expand(B, Z, W).reshape(B * Z, W, 1, 1)
            h = F.relu(h + conv(h) + g)
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
            o = out(F.relu(mix(torch.cat([mx, mn, av], dim=1)))).reshape(B, Z, 2, Y, X)
            score = o[:, :, 0]
            w = torch.softmax(o[:, :, 1], dim=1)
            res.append((w * score).sum(1))
        place = torch.stack(res, dim=1).flatten(1)
        h5 = h.reshape(B, Z, W, Y, X)
        pl = h5.mean((3, 4))
        gin = torch.cat([pl.mean(1), h5.amax((1, 3, 4)), pl.flatten(1)], dim=1)
        g = self.glob(gin)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
