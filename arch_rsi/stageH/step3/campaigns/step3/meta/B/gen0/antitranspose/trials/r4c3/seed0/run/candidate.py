import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
W = 32
D = 64
HEADS = 4
HID = 48


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.np = int(spec.n_place)
        self.T = self.np // (self.Y * self.X)
        N = self.Y * self.X
        cin = int(spec.in_channels) + self.T
        self.stem = nn.Sequential(
            nn.Conv2d(cin, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, W, 1), nn.ReLU())
        self.c1 = nn.Conv2d(W, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.tok = nn.Linear(2 * W, D)
        self.pos = nn.Embedding(N, D)
        self.norm = nn.LayerNorm(D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.aq = nn.Embedding(self.np, D)
        self.qp = nn.Linear(D, D)
        self.m1 = nn.Linear(D, HID)
        self.m2 = nn.Linear(HID, 1)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        hd = D // HEADS
        m = mask[:, :self.np].to(obs.dtype).reshape(B, self.T, Y, X)
        s = self.stem(torch.cat([obs, m], 1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        t = torch.cat([s, f], 1).flatten(2).transpose(1, 2)
        t = self.norm(self.tok(t) + self.pos.weight.unsqueeze(0))
        N = t.shape[1]
        k = self.k(t).view(B, N, HEADS, hd).transpose(1, 2)
        v = self.v(t).view(B, N, HEADS, hd).transpose(1, 2)
        q = self.qp(self.aq.weight).view(1, self.np, HEADS, hd).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, self.np, D)
        o = o + self.aq.weight.unsqueeze(0)
        h = F.relu(self.m1(o))
        place = self.m2(h).squeeze(-1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], -1))
        logits = torch.cat([place, self.done_head(g)], 1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
