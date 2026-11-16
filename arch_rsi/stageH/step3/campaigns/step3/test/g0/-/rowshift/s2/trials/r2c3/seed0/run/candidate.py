import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 8, "n_minibatches": 8, "lr": 0.0007}

D = 48
NH = 4
FF = 96
HH = 32


class Block(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.nh = nh
        self.dh = d // nh
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, ff)
        self.f2 = nn.Linear(ff, d)

    def forward(self, t):
        B, N, Dm = t.shape
        h = self.n1(t)
        qkv = self.qkv(h).view(B, N, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, Dm)
        t = t + self.proj(o)
        t = t + self.f2(F.relu(self.f1(self.n2(t))))
        return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Conv2d(spec.in_channels + 2, D, 3, padding=1)
        self.enc = nn.ModuleList([Block(D, NH, FF) for _ in range(2)])
        self.norm = nn.LayerNorm(D)
        self.h1 = nn.Conv2d(D, HH, 3, padding=1)
        self.h2 = nn.Conv2d(HH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = F.relu(self.stem(torch.cat([obs, yy, xx], dim=1)))
        t = x.flatten(2).transpose(1, 2)
        for layer in self.enc:
            t = layer(t)
        t = self.norm(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        h = F.relu(self.h1(f))
        place = self.h2(h).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
