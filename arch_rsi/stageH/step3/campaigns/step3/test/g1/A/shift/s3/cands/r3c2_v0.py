import math
import torch
import torch.nn as nn
import torch.nn.functional as F

COLW = 96
FEAT = 48
CTX = 48
PROJ = 48
HID = 32
GW = 64
DONE_SCALE = 0.2

RECIPE = {"lr": 0.003, "ppo_epochs": 1, "n_envs": 512, "rollout_len": 16, "n_minibatches": 16, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.3}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels + 2
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.col1 = init_layer(nn.Conv2d(C, COLW, 1))
        self.gn0 = nn.GroupNorm(8, COLW)
        self.col2 = init_layer(nn.Conv2d(COLW, FEAT, 1))
        self.c1 = init_layer(nn.Conv2d(FEAT, CTX, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, CTX)
        self.c2 = init_layer(nn.Conv2d(CTX, CTX, 3, padding=1))
        self.gn2 = nn.GroupNorm(8, CTX)
        self.film = init_layer(nn.Linear(2 * CTX, 2 * CTX), 0.1)
        self.proj = init_layer(nn.Conv2d(FEAT + CTX, PROJ, 1))
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * (FEAT + CTX), GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        c = F.relu(self.gn0(self.col1(torch.cat([obs, yy, xx], 1))))
        c = F.relu(self.col2(c))
        x = F.relu(self.gn1(self.c1(c)))
        x = F.relu(x + self.gn2(self.c2(x)))
        gb = self.film(torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1))
        gam, bet = gb.chunk(2, dim=1)
        x = F.relu(x * (1.0 + gam[:, :, None, None]) + bet[:, :, None, None])
        feats = torch.cat([c, x], 1)
        p = F.relu(self.proj(feats))
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(ex(q))))
        g = self.glob(torch.cat([feats.mean((2, 3)), feats.amax((2, 3))], 1))
        place = torch.cat(res, 1) + self.type_term(g)[:, :, None, None]
        done = -F.softplus(DONE_SCALE * self.done(g))
        logits = torch.cat([place.flatten(1), done], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
