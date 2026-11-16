import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64, "lr": 0.0007, "ent_coef": 0.005, "gamma": 0.97, "gae_lambda": 0.95}

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

    def forward(self, h):
        B, N, _ = h.shape
        dh = D // NH
        q, k, v = self.qkv(self.n1(h)).chunk(3, dim=2)
        q = q.view(B, N, NH, dh).transpose(1, 2)
        k = k.view(B, N, NH, dh).transpose(1, 2)
        v = v.view(B, N, NH, dh).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        h = h + self.o(o)
        h = h + self.f2(F.relu(self.f1(self.n2(h))))
        return h


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        N = int(spec.Y) * int(spec.X)
        self.n_place = int(spec.n_place)
        self.embed = nn.Linear(spec.in_channels, D)
        self.pos = nn.Parameter(torch.randn(1, N, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.aq = nn.Parameter(torch.randn(self.n_place, D) * 0.02)
        self.qp = nn.Linear(D, D)
        self.kp = nn.Linear(D, D)
        self.vp = nn.Linear(D, D)
        self.a1 = nn.Linear(D, 64)
        self.a2 = nn.Linear(64, 1)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        t = obs.permute(0, 2, 3, 1).reshape(B, Y * X, C)
        h = self.embed(t) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.nf(h)
        N = h.shape[1]
        A = self.n_place
        dh = D // NH
        q = self.qp(self.aq).view(A, NH, dh).transpose(0, 1).unsqueeze(0)
        k = self.kp(h).view(B, N, NH, dh).transpose(1, 2)
        v = self.vp(h).view(B, N, NH, dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, A, D)
        place = self.a2(F.relu(self.a1(o + self.aq.unsqueeze(0)))).squeeze(2)
        g = self.glob(torch.cat([h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
