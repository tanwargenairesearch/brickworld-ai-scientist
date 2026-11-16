import math
import torch
import torch.nn as nn
import torch.nn.functional as F

COLW = 96
MID = 48
CTX = 40
RAW = 16
PROJ = 48
HID = 32
HH = 32
DCH = 4
GW = 64
DONE_SCALE = 0.2

RECIPE = {"lr": 0.0015, "ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def cell_ln(ln, x):
    return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = spec.n_types
        self.n_place = spec.n_place
        C = spec.in_channels + spec.n_types + 2
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
        self.film = init_layer(nn.Linear(2 * CTX, 2 * CTX), 0.1)
        self.raw = init_layer(nn.Conv2d(C, RAW, 1))
        self.proj = init_layer(nn.Conv2d(MID + CTX + RAW, PROJ, 1))
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.h3 = init_layer(nn.Conv2d(PROJ, HH, 3, padding=1))
        self.hout = init_layer(nn.Conv2d(HH, spec.n_types, 1), 0.01)
        self.anchor = init_layer(nn.Conv2d(PROJ, 1, 1), 0.01)
        self.dproj = init_layer(nn.Conv2d(PROJ, DCH, 1))
        self.dense = init_layer(nn.Linear(DCH * spec.Y * spec.X, spec.n_place), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * (MID + CTX), GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        m = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.T, Y, X)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        inp = torch.cat([obs, m, yy, xx], 1)
        h = F.relu(cell_ln(self.ln1, self.l1(inp)))
        h = F.relu(h + cell_ln(self.ln2, self.l2(h)))
        h = F.relu(self.l3(h))
        x = F.relu(self.gn1(self.c1(h)))
        x = F.relu(x + self.gn2(self.c2(x)))
        gb = self.film(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        gam, bet = gb.chunk(2, dim=1)
        x = F.relu(x * (1.0 + gam[:, :, None, None]) + bet[:, :, None, None])
        r = self.raw(inp)
        p = F.relu(self.proj(torch.cat([h, x, r], 1)))
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(ex(q))))
        feats = torch.cat([h, x], 1)
        g = self.glob(torch.cat([feats.mean((2, 3)), feats.amax((2, 3))], 1))
        loc = torch.cat(res, 1) + self.hout(F.relu(self.h3(p))) + self.anchor(p) + self.type_term(g)[:, :, None, None]
        d = F.relu(self.dproj(p)).flatten(1)
        place = loc.flatten(1) + self.dense(d)
        done = -F.softplus(DONE_SCALE * self.done(g))
        logits = torch.cat([place, done], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
