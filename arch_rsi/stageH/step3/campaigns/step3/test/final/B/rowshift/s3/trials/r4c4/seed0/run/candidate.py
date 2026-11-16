import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 48
LAYERS = 3
HEADS = 4
FF = 96
GLOB = 64
RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}


class TBlock(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, x):
        B, N, C = x.shape
        hd = C // self.h
        qkv = self.qkv(self.ln1(x)).reshape(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, C)
        x = x + self.o(y)
        return x + self.f2(F.relu(self.f1(self.ln2(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        self.n_actions = int(spec.n_actions)
        cells = self.Y * self.X
        self.K = (self.n_place + cells - 1) // cells
        self.obs_emb = nn.Linear(int(spec.in_channels) + 2, D)
        self.mask_emb = nn.Linear(self.K, D)
        self.pos = nn.Parameter(torch.zeros(1, cells, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TBlock(D, HEADS, FF) for _ in range(LAYERS)])
        self.lnf = nn.LayerNorm(D)
        self.pout = nn.Linear(D, self.K)
        self.glob = nn.Sequential(nn.Linear(2 * D, GLOB), nn.ReLU())
        self.done_head = nn.Linear(GLOB, 1)
        self.gall = nn.Linear(GLOB, self.n_actions)
        self.value_head = nn.Linear(GLOB, 1)
        for m in (self.pout, self.done_head, self.gall):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K = self.Y, self.X, self.K
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        o = torch.cat([obs, yy, xx], dim=1).flatten(2).transpose(1, 2)
        m = mask[:, :self.n_place].to(obs.dtype)
        pad = K * Y * X - self.n_place
        if pad > 0:
            m = F.pad(m, (0, pad))
        m = m.reshape(B, K, Y * X).transpose(1, 2)
        x = self.obs_emb(o) + self.mask_emb(m) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.lnf(x)
        place = self.pout(x).transpose(1, 2).reshape(B, -1)[:, :self.n_place]
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        logits = F.pad(place, (0, 1)) + F.pad(self.done_head(g), (self.n_place, 0)) + self.gall(g)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
