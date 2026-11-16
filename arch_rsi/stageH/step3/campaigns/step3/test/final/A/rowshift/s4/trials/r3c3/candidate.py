import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_envs": 128, "rollout_len": 16, "n_minibatches": 32, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005, "lr": 0.001}

D = 48
HEADS = 4
FF = 96
LAYERS = 2
GLOB = 64
SCALE = 5.0


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.qkv = ortho(nn.Linear(D, 3 * D), 1.0)
        self.o = ortho(nn.Linear(D, D), 1.0)
        self.n2 = nn.LayerNorm(D)
        self.f1 = ortho(nn.Linear(D, FF), math.sqrt(2.0))
        self.f2 = ortho(nn.Linear(FF, D), 1.0)

    def forward(self, x):
        B, N, _ = x.shape
        dh = D // HEADS
        qkv = self.qkv(self.n1(x)).reshape(B, N, 3, HEADS, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh), -1)
        h = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(h)
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        N = spec.Y * spec.X
        self.T = spec.n_types
        self.embed = ortho(nn.Conv2d(spec.in_channels, D, 1), 1.0)
        self.pos = nn.Parameter(torch.randn(N, D))
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D)
        self.pi = ortho(nn.Linear(D, self.T), 0.01)
        self.glob = ortho(nn.Linear(2 * D, GLOB), math.sqrt(2.0))
        self.dense = ortho(nn.Linear(GLOB, spec.n_place), 0.01)
        self.alpha = nn.Parameter(torch.zeros(2))
        self.done = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value = ortho(nn.Linear(GLOB, 1), 1.0)
        self.tbias = ortho(nn.Linear(GLOB, self.T), 0.01)

    def forward(self, obs, mask):
        B = obs.shape[0]
        x = self.embed(obs).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        g = F.relu(self.glob(torch.cat([x.mean(1), x.amax(1)], 1)))
        t = self.pi(x) + self.tbias(g)[:, None, :]
        am = t.reshape(B, -1)
        tm = t.transpose(1, 2).reshape(B, -1)
        w = torch.softmax(SCALE * self.alpha, 0)
        place = w[0] * tm + w[1] * am + self.dense(g)
        logits = torch.cat([place, self.done(g)], 1)
        return logits, self.value(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
