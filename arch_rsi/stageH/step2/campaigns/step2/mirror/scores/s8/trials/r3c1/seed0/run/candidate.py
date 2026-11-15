import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.0005, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

D = 64
NH = 4
FF = 128
NL = 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.n2 = nn.LayerNorm(D)
        self.f1 = nn.Linear(D, FF)
        self.f2 = nn.Linear(FF, D)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(self.n1(x)).view(B, N, 3, NH, D // NH).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D // NH), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.n_place = int(spec.n_place)
        cin = int(spec.in_channels) + self.nt
        self.embed = nn.Linear(cin, D)
        self.row = nn.Parameter(torch.randn(self.Y, D) * 0.02)
        self.col = nn.Parameter(torch.randn(self.X, D) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.place = nn.Linear(D, self.nt)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        mp = mask[:, :self.n_place].to(obs.dtype).reshape(B, self.nt, Y, X)
        x = torch.cat([obs, mp], dim=1).flatten(2).transpose(1, 2)
        pos = (self.row.view(Y, 1, D) + self.col.view(1, X, D)).reshape(1, Y * X, D)
        t = self.embed(x) + pos
        t = torch.cat([self.cls.expand(B, 1, D), t], dim=1)
        for blk in self.blocks:
            t = blk(t)
        t = self.nf(t)
        cell = t[:, 1:]
        place = self.place(cell).transpose(1, 2).reshape(B, -1)
        g = self.glob(torch.cat([t[:, 0], cell.mean(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
