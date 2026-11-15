import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 8, "n_minibatches": 8}

STEM = 64
COL = 48
WIDTH = 40
D = 32
NH = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_cells = int(spec.Y) * int(spec.X)
        self.T = int(spec.n_types)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.tok = nn.Linear(WIDTH, D)
        self.pos = nn.Parameter(torch.zeros(n_cells, D))
        nn.init.normal_(self.pos, std=0.1)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.q = nn.Linear(D, D)
        self.qb = nn.Linear(D, D)
        self.temb = nn.Parameter(torch.zeros(self.T, D))
        nn.init.normal_(self.temb, std=0.1)
        self.o1 = nn.Linear(D, D)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.g2h = nn.Linear(64, D)
        self.o2 = nn.Linear(D, 1)
        nn.init.normal_(self.o2.weight, std=0.01)
        nn.init.zeros_(self.o2.bias)
        self.done_head = nn.Linear(64, 1)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        cells = f.flatten(2).transpose(1, 2)
        t = self.tok(cells) + self.pos.unsqueeze(0)
        N = t.shape[1]
        T = self.T
        dh = D // NH
        K = self.k(t).view(B, N, NH, dh).transpose(1, 2)
        V = self.v(t).view(B, N, NH, dh).transpose(1, 2)
        te = self.temb.view(1, T, 1, D)
        Q = (self.q(t).unsqueeze(1) + te).reshape(B, T * N, NH, dh).transpose(1, 2)
        att = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(dh)
        att = F.softmax(att, dim=-1)
        o = torch.matmul(att, V).transpose(1, 2).reshape(B, T * N, D)
        base = (self.qb(t).unsqueeze(1) + te).reshape(B, T * N, D)
        gg = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.o1(o) + base + self.g2h(gg).unsqueeze(1))
        place = self.o2(h).squeeze(-1)
        logits = torch.cat([place, self.done_head(gg)], dim=1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
