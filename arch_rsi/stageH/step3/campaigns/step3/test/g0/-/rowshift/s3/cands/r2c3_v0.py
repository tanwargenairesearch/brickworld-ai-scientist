import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "lr": 0.0005, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

D = 48
HEADS = 4
FF = 96
DEPTH = 2
DR = 8
DH = 256


class Block(nn.Module):
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
        dh = D // HEADS
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, HEADS, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        a = torch.matmul(att, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(a)
        x = x + self.f2(F.relu(self.f1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        yx = int(spec.Y) * int(spec.X)
        self.embed = nn.Linear(spec.in_channels, D)
        self.pos = nn.Parameter(torch.randn(1, yx, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.lnf = nn.LayerNorm(D)
        self.tok = nn.Linear(D, self.T)
        nn.init.orthogonal_(self.tok.weight, gain=0.01)
        nn.init.zeros_(self.tok.bias)
        self.dred = nn.Linear(D, DR)
        self.dfc = nn.Linear(DR * yx, DH)
        self.dout = nn.Linear(DH, spec.n_place)
        nn.init.zeros_(self.dout.weight)
        nn.init.zeros_(self.dout.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D + DH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        x = obs.flatten(2).transpose(1, 2)
        x = self.embed(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        tl = self.tok(x).transpose(1, 2).reshape(B, -1)
        d = F.relu(self.dred(x)).flatten(1)
        dh = F.relu(self.dfc(d))
        place = tl + self.dout(dh)
        g = self.glob(torch.cat([x.mean(1), x.amax(1), dh], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
