import math
import torch
import torch.nn as nn
import torch.nn.functional as F

W1 = 128
W3 = 64
CTX = 32
PROJ = 48
HID = 32
GW = 64
DONE_SCALE = 0.2

RECIPE = {"lr": 0.0005, "ppo_epochs": 2, "n_envs": 64, "rollout_len": 16, "n_minibatches": 4, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.9, "clip_eps": 0.2}


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def cell_ln(ln, x):
    return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        C = spec.in_channels
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.l1 = init_layer(nn.Conv2d(C, 2 * W1, 1), 1.0)
        self.ln1 = nn.LayerNorm(W1)
        self.l2 = init_layer(nn.Conv2d(W1, W1, 1))
        self.ln2 = nn.LayerNorm(W1)
        self.l3 = init_layer(nn.Conv2d(W1, W3, 1))
        self.c1 = init_layer(nn.Conv2d(W3, CTX, 3, padding=1))
        self.gn1 = nn.GroupNorm(8, CTX)
        self.proj = init_layer(nn.Conv2d(W3 + CTX, PROJ, 1))
        self.exact = nn.ModuleList([init_layer(nn.Conv2d(PROJ, HID, (dy, dx))) for dy, dx in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(PROJ, 1, 1), 0.01)
        self.glob = nn.Sequential(init_layer(nn.Linear(2 * (W3 + CTX), GW)), nn.ReLU())
        self.type_term = init_layer(nn.Linear(GW, spec.n_types), 0.01)
        self.done = init_layer(nn.Linear(GW, 1), 0.01)
        self.value = init_layer(nn.Linear(GW, 1), 1.0)

    def forward(self, obs, mask):
        a, b = self.l1(obs).chunk(2, dim=1)
        h = cell_ln(self.ln1, a * torch.sigmoid(b))
        h = F.relu(h + cell_ln(self.ln2, self.l2(F.relu(h))))
        h = F.relu(self.l3(h))
        x = F.relu(self.gn1(self.c1(h)))
        feats = torch.cat([h, x], 1)
        p = F.relu(self.proj(feats))
        res = []
        for (dy, dx), ex, out in zip(self.fps, self.exact, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(ex(q))))
        g = self.glob(torch.cat([feats.mean((2, 3)), feats.amax((2, 3))], 1))
        place = torch.cat(res, 1) + self.anchor(p) + self.type_term(g)[:, :, None, None]
        done = -F.softplus(DONE_SCALE * self.done(g))
        logits = torch.cat([place.flatten(1), done], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
