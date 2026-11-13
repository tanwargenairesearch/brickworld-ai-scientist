import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 48
HEADS = 4
MLP = 96


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_types = spec.n_types
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, D, 3, padding=1), nn.ReLU(),
            nn.Conv2d(D, D, 3, padding=1), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, spec.Y * spec.X, D))
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, MLP), nn.ReLU(), nn.Linear(MLP, D))
        self.lnf = nn.LayerNorm(D)
        self.place_head = nn.Linear(D, spec.n_types)
        self.glob = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU())
        self.done_head = nn.Linear(D, 1)
        self.value_head = nn.Linear(D, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        B = f.shape[0]
        h = f.flatten(2).transpose(1, 2) + self.pos
        N = h.shape[1]
        dh = D // HEADS
        qkv = self.qkv(self.ln1(h)).view(B, N, 3, HEADS, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, D)
        h = h + self.out(o)
        h = h + self.mlp(self.ln2(h))
        h = self.lnf(h)
        place = self.place_head(h).transpose(1, 2).flatten(1)
        g = self.glob(torch.cat([h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
