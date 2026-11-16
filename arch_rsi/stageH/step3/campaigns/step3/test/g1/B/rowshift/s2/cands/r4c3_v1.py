import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 64, "rollout_len": 8, "n_minibatches": 4, "ppo_epochs": 1, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 48
HP = 32
HK = 3
MEMB = 8
LAYOUTS = [("TYX", ()), ("TXY", ()), ("TYX", (2,)), ("TYX", (3,))]
DILATE = [1, 1, 2]


def to_grid(v, order, flips, T, Y, X):
    B = v.shape[0]
    if order == "TYX":
        g = v.reshape(B, T, Y, X)
    else:
        g = v.reshape(B, T, X, Y).transpose(2, 3)
    if len(flips) > 0:
        g = g.flip(list(flips))
    return g


def from_grid(m, order, flips):
    if len(flips) > 0:
        m = m.flip(list(flips))
    if order == "TYX":
        out = m
    else:
        out = m.transpose(2, 3)
    return out.reshape(m.shape[0], -1)


def coords(obs, Y, X):
    B = obs.shape[0]
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([ys, xs], dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.P = self.Y * self.X
        self.C = int(spec.in_channels)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.grid_ok = (self.n_place == self.T * self.P) and (self.n_actions == self.n_place + 1)
        cin = self.C + 2
        if self.grid_ok:
            cin += len(LAYOUTS) * self.T
        else:
            self.mask_embed = nn.Linear(self.n_actions, MEMB * self.P)
            cin += MEMB
        self.stem1 = nn.Conv2d(cin, STEM, 1)
        self.stem2 = nn.Conv2d(STEM, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in DILATE])
        self.hconv = nn.Conv2d(WIDTH, HP, HK, padding=HK // 2)
        self.hskip = nn.Conv2d(WIDTH, HP, 1)
        if self.grid_ok:
            self.outs = nn.ModuleList([nn.Conv2d(HP, self.T, 1) for _ in LAYOUTS])
        else:
            self.out_proj = nn.Conv2d(HP, 4, 1)
            self.out_lin = nn.Linear(4 * self.P, self.n_actions - 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        m = mask.float()
        parts = [obs, coords(obs, Y, X)]
        if self.grid_ok:
            mp = m[:, :self.n_place]
            for o, fl in LAYOUTS:
                parts.append(to_grid(mp, o, fl, T, Y, X))
        else:
            parts.append(self.mask_embed(m).view(B, MEMB, Y, X))
        x = F.relu(self.stem2(F.relu(self.stem1(torch.cat(parts, dim=1)))))
        for conv in self.res:
            x = F.relu(x + conv(x))
        f = x
        h = F.relu(self.hconv(f) + self.hskip(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        if self.grid_ok:
            place = 0
            for k, (o, fl) in enumerate(LAYOUTS):
                place = place + from_grid(self.outs[k](h), o, fl)
        else:
            place = self.out_lin(self.out_proj(h).flatten(1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
