import math
import torch
import torch.nn as nn
import torch.nn.functional as F

COLW = 96
MID = 48
CTX = 40
PROJ = 32
HID = 32
VW1 = 32
VW2 = 24
GW = 64

RECIPE = {"lr": 0.001, "ppo_epochs": 1, "n_envs": 64, "rollout_len": 64, "n_minibatches": 8, "ent_coef": 0.003, "gamma": 0.995, "gae_lambda": 0.99, "clip_eps": 0.2}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def cell_ln(ln, x):
    return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.l1 = init_layer(nn.Conv2d(C, COLW, 1))
        self.ln1 = nn.LayerNorm(COLW)
        self.l2 = init_layer(nn.Conv2d(COLW, COLW, 1))
        self.ln2 = nn.LayerNorm(COLW)
        self.l3 = init_layer(nn.Conv2d(COLW, MID, 1))
        self.c1 = init_layer(nn.Conv2d(MID, CTX, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, CTX)
        self.c2 = init_layer(nn.Conv2d(CTX, CTX, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, CTX)
        self.proj = init_layer(nn.Conv2d(MID + CTX, PROJ, 1))
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.glob = init_layer(nn.Linear(2 * (MID + CTX), GW))
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.v1 = init_layer(nn.Conv2d(C, VW1, 1))
        self.v2 = init_layer(nn.Conv2d(VW1, VW2, 3, padding=1))
        self.vgn = nn.GroupNorm(8, VW2)
        self.vfc = init_layer(nn.Linear(2 * VW2 + VW1, GW))
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        inp = torch.cat([obs, yy, xx], 1)
        h = F.gelu(cell_ln(self.ln1, self.l1(inp)))
        h = F.gelu(h + cell_ln(self.ln2, self.l2(h)))
        h = F.gelu(self.l3(h))
        x = F.gelu(self.gn1(self.c1(h)))
        x = F.gelu(x + self.gn2(self.c2(x)))
        feats = torch.cat([h, x], 1)
        p = F.gelu(self.proj(feats))
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.gelu(ex(q))))
        g = F.gelu(self.glob(torch.cat([feats.mean((2, 3)), feats.amax((2, 3))], 1)))
        place = torch.cat(res, 1) + self.type_term(g)[:, :, None, None]
        logits = torch.cat([place.flatten(1), self.done(g)], 1)
        v = F.gelu(self.v1(inp))
        v2 = F.gelu(self.vgn(self.v2(v)))
        vg = F.gelu(self.vfc(torch.cat([v2.mean((2, 3)), v2.amax((2, 3)), v.amax((2, 3))], 1)))
        return logits, self.value(vg).squeeze(1)


def build_policy(spec):
    return Net(spec)
