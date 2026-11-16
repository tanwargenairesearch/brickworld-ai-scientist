import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01}

STEM = 64
W = 32
NBLOCK = 2
D = 32
H = 4


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.g = nn.Linear(2 * c, c)

    def forward(self, x):
        gv = self.g(torch.cat([x.mean((2, 3)), x.amax((2, 3))], dim=1))
        h = F.relu(self.c1(x) + gv[:, :, None, None])
        return F.relu(x + self.c2(h))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        self.npl = int(spec.n_place)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.blocks = nn.ModuleList([Block(W) for _ in range(NBLOCK)])
        self.pos = nn.Parameter(torch.randn(1, n, W) * 0.02)
        self.norm = nn.LayerNorm(W)
        self.emb = nn.Parameter(torch.randn(self.npl, D))
        self.wq = nn.Linear(D, D)
        self.wk = nn.Linear(W, D)
        self.wv = nn.Linear(W, D)
        self.wo = nn.Linear(D, D)
        self.we = nn.Linear(D, D)
        self.hid = nn.Linear(D, D)
        self.out = nn.Linear(D, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        f = self.stem(obs)
        for b in self.blocks:
            f = b(f)
        t = self.norm(f.flatten(2).transpose(1, 2) + self.pos)
        N = t.shape[1]
        A = self.npl
        hd = D // H
        q = self.wq(self.emb).view(A, H, hd).permute(1, 0, 2).unsqueeze(0)
        k = self.wk(t).view(B, N, H, hd).permute(0, 2, 1, 3)
        v = self.wv(t).view(B, N, H, hd).permute(0, 2, 1, 3)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        ctx = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(B, A, D)
        h = F.relu(self.wo(ctx) + self.we(self.emb).unsqueeze(0))
        h = F.relu(self.hid(h))
        place = self.out(h).squeeze(-1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
