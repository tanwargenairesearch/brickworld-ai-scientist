import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16, "lr": 0.001, "ent_coef": 0.003}

D = 48
HEADS = 4
MLP = 96
LAYERS = 2


class TBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, MLP), nn.ReLU(), nn.Linear(MLP, D))

    def forward(self, x):
        B, N, _ = x.shape
        hd = D // HEADS
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, HEADS, hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(o)
        x = x + self.mlp(self.ln2(x))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        Y = int(spec.Y)
        X = int(spec.X)
        cin = spec.in_channels + self.nt
        self.embed = nn.Sequential(
            nn.Conv2d(cin, 64, 1), nn.ReLU(), nn.Conv2d(64, D, 1))
        self.pos = nn.Parameter(torch.randn(1, Y * X, D) * 0.02)
        self.layers = nn.ModuleList([TBlock() for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Sequential(
            nn.Conv2d(D, 32, 1), nn.ReLU(),
            nn.Conv2d(32, self.nt, 3, padding=1))
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        m = mask[:, :self.npl].float().view(B, self.nt, Y, X)
        e = self.embed(torch.cat([obs, m], dim=1))
        t = e.flatten(2).transpose(1, 2) + self.pos
        for blk in self.layers:
            t = blk(t)
        t = self.lnf(t)
        grid = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(grid).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
