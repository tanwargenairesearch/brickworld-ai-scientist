import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

D = 64
HEADS = 4
FF = 128
LAYERS = 2


def dihedral(m, k):
    if k >= 4:
        m = m.transpose(2, 3)
    if k % 2 == 1:
        m = m.flip([2])
    if (k // 2) % 2 == 1:
        m = m.flip([3])
    return m


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
        hd = D // HEADS
        qkv = self.qkv(self.n1(x)).reshape(B, N, 3, HEADS, hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        self.n_place = int(spec.n_place)
        self.K = 8 if self.Y == self.X else 4
        n = self.Y * self.X
        cin = int(spec.in_channels) + 2 * self.T
        self.conv = nn.Conv2d(cin, D, 3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, 2 * self.K * self.T)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        T, Y, X = self.T, self.Y, self.X
        mp = mask[:, :self.n_place].to(obs.dtype)
        m1 = mp.reshape(B, T, Y, X)
        m2 = mp.reshape(B, Y, X, T).permute(0, 3, 1, 2)
        h = F.relu(self.conv(torch.cat([obs, m1, m2], dim=1)))
        t = h.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        o = self.out(t).transpose(1, 2).reshape(B, 2, self.K, T, Y, X)
        pa = dihedral(o[:, 0, 0], 0)
        pb = dihedral(o[:, 1, 0], 0)
        for k in range(1, self.K):
            pa = pa + dihedral(o[:, 0, k], k)
            pb = pb + dihedral(o[:, 1, k], k)
        place = pa.flatten(1) + pb.permute(0, 2, 3, 1).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
