import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
FF = 128
HEADS = 4
LAYERS = 2

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.005}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.ff1 = nn.Linear(D, FF)
        self.ff2 = nn.Linear(FF, D)

    def forward(self, x):
        B, N, _ = x.shape
        dh = D // HEADS
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, N, HEADS, dh).transpose(1, 2)
        k = k.reshape(B, N, HEADS, dh).transpose(1, 2)
        v = v.reshape(B, N, HEADS, dh).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.out(o)
        x = x + self.ff2(F.relu(self.ff1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = int(spec.n_types)
        n = int(spec.Y) * int(spec.X)
        self.embed = nn.Linear(int(spec.in_channels), D)
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.place = nn.Linear(D, self.n_types)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        tok = obs.flatten(2).transpose(1, 2)
        x = self.embed(tok) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        place = self.place(x).transpose(1, 2).reshape(B, -1)
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
