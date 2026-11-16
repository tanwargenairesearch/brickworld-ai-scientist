import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 1, "lr": 0.0005, "gamma": 0.95, "gae_lambda": 0.9, "ent_coef": 0.005, "n_minibatches": 16}

D = 32
NH = 4
FF = 64
NL = 2
G = 64


class Layer(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.nh = nh
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.nh
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.reshape(B, N, self.nh, hd).transpose(1, 2)
        k = k.reshape(B, N, self.nh, hd).transpose(1, 2)
        v = v.reshape(B, N, self.nh, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nT = int(spec.n_types)
        Y, X = int(spec.Y), int(spec.X)
        cin = int(spec.in_channels) + self.nT
        self.e1 = nn.Conv2d(cin, 64, 1)
        self.e2 = nn.Conv2d(64, D, 1)
        self.pos = nn.Parameter(torch.randn(1, D, Y, X) * 0.02)
        self.loc = nn.Conv2d(D, D, 3, padding=1)
        self.layers = nn.ModuleList([Layer(D, NH, FF) for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.hconv = nn.Conv2d(2 * D, D, 1)
        self.out = nn.Conv2d(D, self.nT, 1)
        nn.init.orthogonal_(self.out.weight, 0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        m = mask[:, :self.nT * Y * X].to(obs.dtype).reshape(B, self.nT, Y, X)
        s = F.relu(self.e1(torch.cat([obs, m], dim=1)))
        s = self.e2(s) + self.pos
        s = s + F.relu(self.loc(F.relu(s)))
        t = s.flatten(2).transpose(1, 2)
        for layer in self.layers:
            t = layer(t)
        t = self.nf(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        u = F.relu(self.hconv(torch.cat([f, F.relu(s)], dim=1)))
        place = self.out(u).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
