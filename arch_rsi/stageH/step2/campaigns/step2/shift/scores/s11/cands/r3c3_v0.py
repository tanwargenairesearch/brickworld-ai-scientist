import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "n_envs": 64, "rollout_len": 16, "n_minibatches": 8}

STEM = 64
COL = 48
WIDTH = 40
D = 64
NH = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n_cells = int(spec.Y) * int(spec.X)
        self.A = int(spec.n_actions)
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
        self.qemb = nn.Embedding(self.A, D)
        self.qp = nn.Linear(D, D)
        self.o1 = nn.Linear(D, D)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, 64), nn.ReLU())
        self.g2h = nn.Linear(64, D)
        self.o2 = nn.Linear(D, 1)
        nn.init.normal_(self.o2.weight, std=0.01)
        nn.init.zeros_(self.o2.bias)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        s = self.stem(obs)
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        cells = f.flatten(2).transpose(1, 2)
        t = self.tok(cells) + self.pos.unsqueeze(0)
        N = t.shape[1]
        dh = D // NH
        K = self.k(t).view(B, N, NH, dh).transpose(1, 2)
        V = self.v(t).view(B, N, NH, dh).transpose(1, 2)
        Q = self.qp(self.qemb.weight).view(self.A, NH, dh).transpose(0, 1)
        att = torch.einsum('had,bhnd->bhan', Q, K) / math.sqrt(dh)
        att = F.softmax(att, dim=-1)
        o = torch.einsum('bhan,bhnd->bhad', att, V)
        o = o.transpose(1, 2).reshape(B, self.A, D)
        gg = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.o1(o) + self.g2h(gg).unsqueeze(1))
        logits = self.o2(h).squeeze(-1)
        return logits, self.value_head(gg).squeeze(1)


def build_policy(spec):
    return Net(spec)
