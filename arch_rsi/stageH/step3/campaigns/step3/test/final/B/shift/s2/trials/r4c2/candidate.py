import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 2e-3, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.85}

STEM = 64
D = 40
HEADS = 4
HEADW = 32


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 2 * d)
        self.f2 = nn.Linear(2 * d, d)

    def forward(self, x):
        B, N, d = x.shape
        H = self.heads
        dh = d // H
        qkv = self.qkv(self.ln1(x)).reshape(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(B, N, d)
        x = x + self.proj(y)
        x = x + self.f2(F.relu(self.f1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.n_place = int(spec.n_place)
        hw = self.Y * self.X
        self.hw = hw
        self.K = (self.n_place + hw - 1) // hw
        K = self.K
        self.stem = nn.Sequential(nn.Conv2d(int(spec.in_channels) + 2 + K, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, D, 1), nn.ReLU())
        self.r1 = nn.Conv2d(D, D, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, hw, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blk = Block(D, HEADS)
        self.r2 = nn.Conv2d(D, D, 3, padding=1)
        self.h3 = nn.Conv2d(D, HEADW, 3, padding=1)
        self.h1 = nn.Conv2d(D, HEADW, 1)
        self.hout = nn.Conv2d(HEADW, K, 1)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X, K, hw = self.Y, self.X, self.K, self.hw
        mp = mask[:, :self.n_place].to(obs.dtype)
        if K * hw > self.n_place:
            mp = F.pad(mp, (0, K * hw - self.n_place))
        mg = mp.reshape(B, K, Y, X)
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat([obs, ys, xs, mg], 1))
        h = F.relu(h + self.r1(h))
        t = h.reshape(B, D, hw).transpose(1, 2) + self.pos
        t = self.blk(t)
        h = t.transpose(1, 2).reshape(B, D, Y, X)
        h = F.relu(h + self.r2(h))
        a = F.relu(self.h3(h) + self.h1(h))
        o = self.hout(a)
        place = o.reshape(B, K * hw)[:, :self.n_place]
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
