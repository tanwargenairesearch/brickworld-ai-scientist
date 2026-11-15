import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "gamma": 0.97, "ent_coef": 0.005}

D = 64
H = 4
L = 2
FF = 128
DPROJ = 16
GH = 64


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, FF), nn.ReLU(), nn.Linear(FF, D))

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(self.ln1(x)).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D // H), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(y)
        return x + self.ff(self.ln2(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.embed = nn.Linear(spec.in_channels, D)
        self.pos = nn.Parameter(torch.zeros(1, n, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.tok_head = nn.Linear(D, spec.n_types)
        self.dproj = nn.Linear(D, DPROJ)
        self.dense = nn.Linear(DPROJ * n, spec.n_actions)
        nn.init.normal_(self.dense.weight, std=0.01)
        nn.init.zeros_(self.dense.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, GH), nn.ReLU())
        self.done_head = nn.Linear(GH, 1)
        self.value_head = nn.Linear(GH, 1)

    def forward(self, obs, mask):
        x = obs.flatten(2).transpose(1, 2)
        h = self.embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.lnf(h)
        place = self.tok_head(h).transpose(1, 2).flatten(1)
        g = self.glob(torch.cat([h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        d = self.dense(F.relu(self.dproj(h)).flatten(1))
        logits = logits + d
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
