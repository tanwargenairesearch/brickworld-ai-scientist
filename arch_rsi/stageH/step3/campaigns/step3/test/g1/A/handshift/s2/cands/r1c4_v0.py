import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {'lr': 0.001, 'n_envs': 128, 'rollout_len': 32, 'n_minibatches': 16, 'ppo_epochs': 4, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.01}

D = 12
E = 16
HID = 48
CTX = 32
G = 64
PER = 9


def ortho(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        Z = int(spec.Z)
        self.Z = Z
        T = len(self.fps)
        self.enc = ortho(nn.Conv2d(PER * Z, D * Z, 1, groups=Z))
        self.enc2 = ortho(nn.Conv2d(D * Z, D * Z, 1, groups=Z), 1.0)
        self.ctx = ortho(nn.Conv2d(D * Z, CTX, 3, padding=1))
        self.ctxn = nn.GroupNorm(4, CTX)
        self.lay = nn.ModuleList([ortho(nn.Conv2d(2 * D * Z, E * Z, 1, groups=Z)) for t in range(T)])
        self.cross = nn.ModuleList([ortho(nn.Conv2d(E * Z + CTX, HID, 1)) for t in range(T)])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(HID, 1, 1), 0.01) for t in range(T)])
        self.anchor = ortho(nn.Conv2d(D * Z + CTX, 1, 1), 0.01)
        self.type_bias = nn.Parameter(torch.zeros(T))
        self.glob = nn.Sequential(ortho(nn.Linear(2 * (D * Z + CTX), G)), nn.ReLU(), ortho(nn.Linear(G, G)), nn.ReLU())
        self.done = ortho(nn.Linear(G, 1), 0.01)
        self.vnet = nn.Sequential(ortho(nn.Linear(2 * (D * Z + CTX), G)), nn.ReLU(), ortho(nn.Linear(G, 1), 1.0))

    def forward(self, obs, mask):
        Z = self.Z
        B, Cin, Y, X = obs.shape
        zero = torch.zeros(B, 1, Y, X, device=obs.device, dtype=obs.dtype)
        glob = obs[:, 4 * Z:4 * Z + 3]
        parts = []
        for z in range(Z):
            if z > 0:
                below = obs[:, z - 1:z]
            else:
                below = zero
            pos = torch.full((B, 1, Y, X), (z + 1.0) / Z, device=obs.device, dtype=obs.dtype)
            parts += [obs[:, z:z + 1], obs[:, Z + z:Z + z + 1], obs[:, 2 * Z + z:2 * Z + z + 1],
                      obs[:, 3 * Z + z:3 * Z + z + 1], below, glob, pos]
        x = torch.cat(parts, dim=1)
        e = F.relu(self.enc(x))
        e = F.relu(e + self.enc2(e))
        c = F.relu(self.ctxn(self.ctx(e)))
        h = torch.cat([e, c], dim=1)
        a = self.anchor(h)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            if dy * dx == 1:
                mx, mn = e, e
            else:
                q = F.pad(e, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
            u = torch.cat([mx.reshape(B, Z, D, Y, X), mn.reshape(B, Z, D, Y, X)], dim=2).reshape(B, 2 * D * Z, Y, X)
            l = F.relu(self.lay[i](u))
            z2 = F.relu(self.cross[i](torch.cat([l, c], dim=1)))
            res.append(self.outs[i](z2) + a + self.type_bias[i])
        place = torch.cat(res, dim=1).flatten(1)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        g = self.glob(pooled)
        logits = torch.cat([place, self.done(g)], dim=1)
        return logits, self.vnet(pooled).squeeze(1)


def build_policy(spec):
    return Net(spec)
