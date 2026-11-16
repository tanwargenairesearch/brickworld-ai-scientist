import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "ent_coef": 0.03, "gamma": 0.99, "gae_lambda": 0.95}

W = 32
D = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.T = int(spec.n_types)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.A = int(spec.n_place)
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(nn.Conv2d(cin, 48, 1), nn.ReLU(), nn.Conv2d(48, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.key = nn.Linear(W, D)
        self.val = nn.Linear(W, D)
        self.pos = nn.Embedding(self.Y * self.X, D)
        self.qry = nn.Embedding(self.A, D)
        self.abias = nn.Parameter(torch.zeros(self.A, D))
        self.out = nn.Linear(D, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.scale = 1.0 / math.sqrt(D)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        m = mask[:, :self.A].to(obs.dtype).reshape(B, self.T, self.Y, self.X)
        s = self.stem(torch.cat([obs, m], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        cells = f.flatten(2).transpose(1, 2)
        k = self.key(cells) + self.pos.weight.unsqueeze(0)
        v = self.val(cells)
        q = self.qry.weight
        att = torch.matmul(k, q.t()).transpose(1, 2) * self.scale
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v)
        h = F.relu(o + self.abias.unsqueeze(0))
        place = self.out(h).squeeze(-1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
