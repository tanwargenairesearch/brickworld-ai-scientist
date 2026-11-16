import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2}

STEM = 64
COL = 48
WIDTH = 40
D = 32
NH = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.N = int(spec.Y) * int(spec.X)
        self.A = int(spec.n_place)
        self.hd = D // NH
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.kp = nn.Linear(WIDTH, D)
        self.vp = nn.Linear(WIDTH, D)
        self.pos = nn.Parameter(torch.randn(1, self.N, D))
        self.query = nn.Parameter(torch.randn(self.A, D))
        self.aemb = nn.Parameter(torch.zeros(self.A, D))
        self.o1 = nn.Linear(D, D)
        self.o2 = nn.Linear(D, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        tok = f.flatten(2).transpose(1, 2)
        k = (self.kp(tok) + self.pos).reshape(B, self.N, NH, self.hd).permute(0, 2, 1, 3)
        v = self.vp(tok).reshape(B, self.N, NH, self.hd).permute(0, 2, 1, 3)
        q = self.query.reshape(self.A, NH, self.hd).permute(1, 0, 2).unsqueeze(0)
        att = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.hd)
        att = torch.softmax(att, dim=-1)
        u = torch.matmul(att, v)
        u = u.permute(0, 2, 1, 3).reshape(B, self.A, D)
        h = F.relu(self.o1(u) + self.aemb)
        place = self.o2(h).squeeze(2)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
