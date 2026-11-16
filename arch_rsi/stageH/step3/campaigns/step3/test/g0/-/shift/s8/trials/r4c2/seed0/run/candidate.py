import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}

STEM = 64
COL = 48
WIDTH = 40
D = 48
H = 6
OH = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.yy = int(spec.Y)
        self.xx = int(spec.X)
        self.nt = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.k = nn.Linear(WIDTH, D)
        self.v = nn.Linear(WIDTH, D)
        self.krow = nn.Parameter(torch.randn(self.yy, D) * 0.7)
        self.kcol = nn.Parameter(torch.randn(self.xx, D) * 0.7)
        self.qt = nn.Parameter(torch.randn(self.nt, D) * 0.5)
        self.qr = nn.Parameter(torch.randn(self.yy, D) * 0.7)
        self.qc = nn.Parameter(torch.randn(self.xx, D) * 0.7)
        self.o1 = nn.Linear(D, OH)
        self.qb = nn.Linear(D, OH)
        self.gq = nn.Linear(64, OH)
        self.o2 = nn.Linear(OH, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        b = obs.shape[0]
        n = self.yy * self.xx
        p = self.nt * n
        hd = D // H
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        tok = f.flatten(2).transpose(1, 2)
        kpos = (self.krow[:, None, :] + self.kcol[None, :, :]).reshape(n, D)
        k = self.k(tok) + kpos.unsqueeze(0)
        v = self.v(tok)
        q = (self.qt[:, None, None, :] + self.qr[None, :, None, :] + self.qc[None, None, :, :]).reshape(p, D)
        kh = k.reshape(b, n, H, hd).permute(0, 2, 1, 3)
        vh = v.reshape(b, n, H, hd).permute(0, 2, 1, 3)
        qh = q.reshape(p, H, hd).permute(1, 0, 2).unsqueeze(0)
        att = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        o = torch.matmul(att, vh).permute(0, 2, 1, 3).reshape(b, p, D)
        h = F.relu(self.o1(o) + self.qb(q).unsqueeze(0) + self.gq(g).unsqueeze(1))
        place = self.o2(h).squeeze(-1)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
