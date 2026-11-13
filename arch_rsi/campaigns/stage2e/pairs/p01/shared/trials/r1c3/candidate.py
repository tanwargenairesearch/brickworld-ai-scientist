import math
import torch
import torch.nn as nn
import torch.nn.functional as F

W = 48
HEADS = 4
RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 4, "gamma": 0.97, "lr": 0.0007}


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(spec.in_channels + 2, W, 1), nn.ReLU(),
                                  nn.Conv2d(W, W, 3, padding=1), nn.ReLU())
        self.ln1 = nn.LayerNorm(W)
        self.qkv = nn.Linear(W, 3 * W)
        self.proj = nn.Linear(W, W)
        self.ln2 = nn.LayerNorm(W)
        self.mlp = nn.Sequential(nn.Linear(W, 2 * W), nn.ReLU(), nn.Linear(2 * W, W))
        self.foot = nn.Conv2d(W, W, 2)
        self.place_head = nn.Conv2d(W, spec.n_types, 1)
        nn.init.orthogonal_(self.place_head.weight, 0.01)
        nn.init.zeros_(self.place_head.bias)
        self.glob = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU())
        self.done_head = nn.Linear(W, 1)
        nn.init.orthogonal_(self.done_head.weight, 0.01)
        nn.init.zeros_(self.done_head.bias)
        self.value_head = nn.Linear(W, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        yy = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xx = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        h = self.stem(torch.cat([obs, yy, xx], dim=1))
        N = Y * X
        t = h.flatten(2).transpose(1, 2)
        u = self.ln1(t)
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        d = W // HEADS
        q = q.reshape(B, N, HEADS, d).transpose(1, 2)
        k = k.reshape(B, N, HEADS, d).transpose(1, 2)
        v = v.reshape(B, N, HEADS, d).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, W)
        t = t + self.proj(o)
        t = t + self.mlp(self.ln2(t))
        h = t.transpose(1, 2).reshape(B, W, Y, X)
        f = F.relu(self.foot(F.pad(h, (0, 1, 0, 1))))
        place = self.place_head(f).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
