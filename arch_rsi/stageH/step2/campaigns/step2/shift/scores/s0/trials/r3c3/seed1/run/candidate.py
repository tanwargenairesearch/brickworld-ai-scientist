import torch
import torch.nn as nn
import torch.nn.functional as F
import math

RECIPE = {"ppo_epochs": 1, "n_envs": 128, "rollout_len": 16, "n_minibatches": 16,
          "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.005}

W = 40
D = 32
NH = 2
HQ = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.A = int(spec.n_actions)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 48, 1), nn.ReLU())
        self.c1 = nn.Conv2d(48, W, 3, padding=1)
        self.c2 = nn.Conv2d(W, W, 3, padding=1)
        self.kv = nn.Linear(W, 2 * D)
        self.q = nn.Embedding(self.A, D)
        self.qmix = nn.Linear(D, HQ)
        self.amix = nn.Linear(D, HQ)
        self.gmix = nn.Linear(64, HQ)
        self.out = nn.Linear(HQ, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, 64), nn.ReLU())
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        s = self.stem(torch.cat([obs, yy, xx], dim=1))
        x = F.relu(self.c1(s))
        f = F.relu(x + self.c2(x))
        N = Y * X
        tok = f.flatten(2).transpose(1, 2)
        kv = self.kv(tok)
        k, v = kv.chunk(2, dim=-1)
        dh = D // NH
        k = k.reshape(B, N, NH, dh).transpose(1, 2)
        v = v.reshape(B, N, NH, dh).transpose(1, 2)
        qw = self.q.weight
        q = qw.reshape(self.A, NH, dh).transpose(0, 1).unsqueeze(0)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(att, v)
        o = o.transpose(1, 2).reshape(B, self.A, D)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        h = F.relu(self.amix(o) + self.qmix(qw).unsqueeze(0) + self.gmix(g).unsqueeze(1))
        logits = self.out(h).squeeze(-1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
