import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8,
          "lr": 0.0007, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.003}

D = 48
NH = 4
FF = 96
NL = 2
HH = 64


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.f1 = nn.Linear(D, FF)
        self.f2 = nn.Linear(FF, D)

    def forward(self, x):
        B, N, _ = x.shape
        dh = D // NH
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, N, NH, dh).transpose(1, 2)
        k = k.reshape(B, N, NH, dh).transpose(1, 2)
        v = v.reshape(B, N, NH, dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        a = torch.matmul(att, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(a)
        x = x + self.f2(F.relu(self.f1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.ky = max(a for a, b in fps)
        self.kx = max(b for a, b in fps)
        T = int(spec.n_types)
        N = int(spec.Y) * int(spec.X)
        self.embed = nn.Conv2d(spec.in_channels, D, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, N, D))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([Layer() for _ in range(NL)])
        self.lnf = nn.LayerNorm(D)
        self.win = nn.Conv2d(D, HH, (self.ky, self.kx))
        self.out = nn.Conv2d(HH, T, 1)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        e = self.embed(obs)
        t = e.flatten(2).transpose(1, 2) + self.pos
        for layer in self.layers:
            t = layer(t)
        t = self.lnf(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        q = F.pad(f, (0, self.kx - 1, 0, self.ky - 1))
        h = F.relu(self.win(q))
        place = self.out(h).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
