import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 7e-4, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.01}

D = 64
NH = 4
FF = 128
NL = 2
P = 32
H = 32


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
        hd = D // NH
        qkv = self.qkv(self.n1(x)).reshape(B, N, 3, NH, hd).permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(y)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.embed = nn.Sequential(nn.Linear(int(spec.in_channels), D), nn.ReLU(), nn.Linear(D, D))
        self.pos = nn.Parameter(torch.randn(self.Y * self.X, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.nf = nn.LayerNorm(D)
        self.proj = nn.Linear(D, P)
        self.mix = nn.ModuleList([nn.Conv2d(3 * P, H, 1) for _ in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(H, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        t = obs.flatten(2).transpose(1, 2)
        x = self.embed(t) + self.pos.unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        x = self.nf(x)
        p = self.proj(x).transpose(1, 2).reshape(B, P, self.Y, self.X)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            if dy * dx == 1:
                mx, mn, av = p, p, p
            else:
                q = F.pad(p, (0, dx - 1, 0, dy - 1))
                mx = F.max_pool2d(q, (dy, dx), stride=1)
                mn = -F.max_pool2d(-q, (dy, dx), stride=1)
                av = F.avg_pool2d(q, (dy, dx), stride=1)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
