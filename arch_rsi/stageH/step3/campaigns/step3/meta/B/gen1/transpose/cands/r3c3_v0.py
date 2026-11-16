import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "ppo_epochs": 3, "ent_coef": 0.002, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
WIDTH = 40
HEAD = 32
N_RES = 2


def transpose_perm(T, Y, X):
    t = torch.arange(T).view(T, 1, 1).expand(T, Y, X)
    y = torch.arange(Y).view(1, Y, 1).expand(T, Y, X)
    x = torch.arange(X).view(1, 1, X).expand(T, Y, X)
    idx = t * (Y * X) + x * Y + y
    return idx.reshape(-1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        T, Y, X = int(spec.n_types), int(spec.Y), int(spec.X)
        self.T, self.Y, self.X = T, Y, X
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        self.grid_ok = (self.n_place == T * Y * X) and (self.n_actions == self.n_place + 1)
        C = int(spec.in_channels)
        yy = torch.linspace(-1.0, 1.0, Y).view(1, 1, Y, 1).expand(1, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X).view(1, 1, 1, X).expand(1, 1, Y, X)
        self.register_buffer("coords", torch.cat([yy, xx], dim=1).contiguous())
        in_ch = C + 2
        if self.grid_ok:
            p = transpose_perm(T, Y, X)
            self.register_buffer("perm", p)
            self.register_buffer("inv", torch.argsort(p))
            in_ch += T
        self.s1 = nn.Conv2d(in_ch, STEM, 1)
        self.s2 = nn.Conv2d(STEM, WIDTH, 1)
        self.res = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 3, padding=1) for _ in range(N_RES)])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        if self.grid_ok:
            self.hk = nn.Conv2d(WIDTH, HEAD, 3, padding=1)
            self.h1 = nn.Conv2d(WIDTH, HEAD, 1)
            self.hout = nn.Conv2d(HEAD, T, 1)
        else:
            self.membed = nn.Linear(self.n_actions, STEM)
            self.fb_out = nn.Linear(64, self.n_actions)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, T = self.Y, self.X, self.T
        parts = [obs, self.coords.expand(B, -1, -1, -1).to(obs.dtype)]
        m = mask.to(obs.dtype)
        if self.grid_ok:
            parts.append(m[:, :self.n_place].index_select(1, self.perm).view(B, T, Y, X))
        s = self.s1(torch.cat(parts, dim=1))
        if not self.grid_ok:
            s = s + self.membed(m).view(B, -1, 1, 1)
        f = F.relu(self.s2(F.relu(s)))
        for conv in self.res:
            f = F.relu(f + conv(f))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        done = self.done_head(g)
        value = self.value_head(g).squeeze(1)
        if not self.grid_ok:
            return self.fb_out(g), value
        h = F.relu(self.hk(f) + self.h1(f))
        maps = self.hout(h).reshape(B, -1)
        place = maps.index_select(1, self.inv) * 2.0
        return torch.cat([place, done], dim=1), value


def build_policy(spec):
    return Net(spec)
