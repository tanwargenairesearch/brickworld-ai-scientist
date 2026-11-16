import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_minibatches": 16, "gamma": 0.98, "gae_lambda": 0.95, "lr": 0.001, "ent_coef": 0.01}

W = 24
P = 16
HID = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Z = int(spec.Z)
        self.T = int(spec.n_types)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.register_buffer("pos", torch.eye(self.Z).reshape(1, self.Z, self.Z, 1, 1))
        cin = 12 + 3 + self.Z
        self.inp = nn.Conv2d(cin, W, 1)
        self.zmix = nn.Conv2d(3 * W, W, 1)
        self.glob1 = nn.Linear(2 * W, W)
        self.c3 = nn.Conv2d(W, W, 3, padding=1)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, P, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(P, 1, 1) for _ in self.fps])
        self.g = nn.Sequential(nn.Linear(2 * W, HID), nn.ReLU())
        self.done = nn.Linear(HID, 1)
        self.value = nn.Linear(HID, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Z = self.Z
        Y = obs.shape[2]
        X = obs.shape[3]
        per = obs[:, :4 * Z].reshape(B, 4, Z, Y, X).transpose(1, 2)
        zero = torch.zeros_like(per[:, :1])
        below = torch.cat([zero, per[:, :-1]], 1)
        above = torch.cat([per[:, 1:], zero], 1)
        gl = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        pos = self.pos.expand(B, Z, Z, Y, X)
        x = torch.cat([per, below, above, gl, pos], 2).reshape(B * Z, -1, Y, X)
        h = F.relu(self.inp(x))
        h5 = h.reshape(B, Z, W, Y, X)
        up = torch.cummax(h5, 1).values
        dn = torch.cummax(h5.flip(1), 1).values.flip(1)
        h = F.relu(self.zmix(torch.cat([h5, up, dn], 2).reshape(B * Z, 3 * W, Y, X)))
        h5 = h.reshape(B, Z, W, Y, X)
        pooled = torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4))], 1)
        h5 = F.relu(h5 + self.glob1(pooled)[:, None, :, None, None])
        h = h5.reshape(B * Z, W, Y, X)
        h = F.relu(h + self.c3(h))
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
            hh = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(hh))
        s = torch.cat(res, 1).reshape(B, Z, self.T, Y, X)
        place = torch.logsumexp(s, 1).flatten(1)
        h5 = h.reshape(B, Z, W, Y, X)
        g = self.g(torch.cat([h5.mean((1, 3, 4)), h5.amax((1, 3, 4))], 1))
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
