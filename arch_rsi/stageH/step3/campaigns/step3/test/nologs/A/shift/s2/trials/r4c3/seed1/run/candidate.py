import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 1, "lr": 0.0015, "gamma": 0.97, "gae_lambda": 0.95, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16}

STEM = 48
D = 32
GLOB = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = int(spec.Y) * int(spec.X)
        na = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 3, padding=1), nn.ReLU())
        self.c2 = nn.Conv2d(D, D, 3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.1)
        self.act = nn.Parameter(torch.randn(na, D) * 0.1)
        self.wq = nn.Linear(D, D)
        self.wk = nn.Linear(D, D)
        self.wv = nn.Linear(D, D)
        self.glob = nn.Sequential(nn.Linear(2 * D, GLOB), nn.ReLU())
        self.gp = nn.Linear(GLOB, D)
        self.mixh = nn.Linear(D, D)
        self.out = nn.Linear(D, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.value_head = nn.Linear(GLOB, 1)

    def forward(self, obs, mask):
        s = self.stem(obs)
        f = F.relu(s + self.c2(s))
        t = f.flatten(2).transpose(1, 2) + self.pos
        k = self.wk(t)
        v = self.wv(t)
        q = self.wq(self.act)
        sc = torch.matmul(k, q.transpose(0, 1)).transpose(1, 2) / math.sqrt(D)
        a = torch.softmax(sc, dim=-1)
        r = torch.matmul(a, v)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        h = F.relu(r + self.act.unsqueeze(0) + self.gp(g).unsqueeze(1))
        h = F.relu(self.mixh(h))
        logits = self.out(h).squeeze(-1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
