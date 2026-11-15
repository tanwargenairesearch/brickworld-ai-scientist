import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 1, "n_minibatches": 8, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.01}

D = 64
NB = 3


class MixBlock(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.t1 = nn.Linear(n, n)
        self.t2 = nn.Linear(n, n)
        self.ln2 = nn.LayerNorm(d)
        self.c1 = nn.Linear(d, 2 * d)
        self.c2 = nn.Linear(2 * d, d)

    def forward(self, x):
        y = self.ln1(x).transpose(1, 2)
        y = self.t2(F.gelu(self.t1(y))).transpose(1, 2)
        x = x + y
        x = x + self.c2(F.gelu(self.c1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.nt = int(spec.n_types)
        self.embed = nn.Linear(spec.in_channels, D)
        self.pos = nn.Parameter(torch.zeros(1, n, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([MixBlock(n, D) for _ in range(NB)])
        self.ln = nn.LayerNorm(D)
        self.place = nn.Linear(D, self.nt)
        self.omix = nn.Linear(n, n)
        nn.init.normal_(self.omix.weight, std=0.01)
        nn.init.zeros_(self.omix.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        t = obs.flatten(2).transpose(1, 2)
        x = self.embed(t) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        p = self.place(x).transpose(1, 2)
        p = p + self.omix(p)
        place = p.flatten(1)
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
