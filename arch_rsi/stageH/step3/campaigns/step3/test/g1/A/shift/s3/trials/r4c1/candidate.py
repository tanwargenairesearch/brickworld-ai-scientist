import math
import torch
import torch.nn as nn
import torch.nn.functional as F

COLW = 96
MID = 48
CTX = 40
PROJ = 40
HH = 32
GW = 64

RECIPE = {"lr": 0.001, "ppo_epochs": 2, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}


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
        self.Y = spec.Y
        self.X = spec.X
        self.n_place = spec.n_place
        C = spec.in_channels + 4 * spec.n_types + 2
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
        self.h3 = init_layer(nn.Conv2d(PROJ, HH, 3, padding=1))
        self.hout = init_layer(nn.Conv2d(HH, spec.n_types, 1), 0.01)
        self.gate = nn.Parameter(torch.zeros(4))
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * (MID + CTX), GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def mask_layouts(self, m, B):
        T, Y, X = self.T, self.Y, self.X
        a = m.reshape(B, T, Y, X)
        b = m.reshape(B, Y, X, T).permute(0, 3, 1, 2)
        c = m.reshape(B, T, X, Y).permute(0, 1, 3, 2)
        d = m.reshape(B, X, Y, T).permute(0, 3, 2, 1)
        return torch.cat([a, b, c, d], 1)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        m = self.mask_layouts(mask[:, :self.n_place].to(obs.dtype), B)
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        inp = torch.cat([obs, m, yy, xx], 1)
        h = F.relu(cell_ln(self.ln1, self.l1(inp)))
        h = F.relu(h + cell_ln(self.ln2, self.l2(h)))
        h = F.relu(self.l3(h))
        x = F.relu(self.gn1(self.c1(h)))
        x = F.relu(x + self.gn2(self.c2(x)))
        feats = torch.cat([h, x], 1)
        p = F.relu(self.proj(feats))
        g = self.glob(torch.cat([feats.mean((2, 3)), feats.amax((2, 3))], 1))
        S = self.hout(F.relu(self.h3(p))) + self.type_term(g)[:, :, None, None]
        w = torch.softmax(self.gate, 0)
        s0 = S.flatten(1)
        s1 = S.permute(0, 2, 3, 1).flatten(1)
        s2 = S.permute(0, 1, 3, 2).flatten(1)
        s3 = S.permute(0, 3, 2, 1).flatten(1)
        place = w[0] * s0 + w[1] * s1 + w[2] * s2 + w[3] * s3
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
