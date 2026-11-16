import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LF1 = 32
LF = 16
MIX = 48
CTX = 40
PROJ = 32
HID = 32
GW = 64

RECIPE = {"lr": 0.001, "ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.Z = spec.Z
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.e1 = init_layer(nn.Conv2d(10, LF1, 1))
        self.e2 = init_layer(nn.Conv2d(LF1, LF, 1))
        self.mix = init_layer(nn.Conv2d(spec.Z * LF + C, MIX, 1))
        self.gn0 = nn.GroupNorm(8, MIX)
        self.c1 = init_layer(nn.Conv2d(MIX, CTX, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, CTX)
        self.c2 = init_layer(nn.Conv2d(CTX, CTX, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, CTX)
        self.proj = init_layer(nn.Conv2d(MIX + CTX, PROJ, 1))
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * (MIX + CTX), GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        Z = self.Z
        bld = obs[:, 0:Z]
        tgt = obs[:, Z:2 * Z]
        mis = obs[:, 2 * Z:3 * Z]
        ext = obs[:, 3 * Z:4 * Z]
        below = F.pad(bld, (0, 0, 0, 0, 1, 0), value=1.0)[:, :Z]
        above = F.pad(tgt, (0, 0, 0, 0, 0, 1))[:, 1:]
        per = torch.stack([bld, tgt, mis, ext, below, above], 2)
        g3 = obs[:, 4 * Z:4 * Z + 3].unsqueeze(1).expand(B, Z, 3, Y, X)
        zc = torch.linspace(0.0, 1.0, Z, device=obs.device, dtype=obs.dtype).view(1, Z, 1, 1, 1).expand(B, Z, 1, Y, X)
        inp = torch.cat([per, g3, zc], 2).reshape(B * Z, 10, Y, X)
        e = F.relu(self.e1(inp))
        e = F.relu(self.e2(e)).reshape(B, Z * LF, Y, X)
        s = F.relu(self.gn0(self.mix(torch.cat([e, obs], 1))))
        x = F.relu(self.gn1(self.c1(s)))
        x = F.relu(x + self.gn2(self.c2(x)))
        feats = torch.cat([s, x], 1)
        p = F.relu(self.proj(feats))
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(ex(q))))
        g = self.glob(torch.cat([feats.mean((2, 3)), feats.amax((2, 3))], 1))
        place = torch.cat(res, 1) + self.type_term(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
