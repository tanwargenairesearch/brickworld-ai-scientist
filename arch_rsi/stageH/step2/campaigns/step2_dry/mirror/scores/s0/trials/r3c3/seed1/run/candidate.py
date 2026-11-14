import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.98, "gae_lambda": 0.95}

STEM = 64
D = 48
HEADS = 4
HEAD = 48


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, t):
        h = self.n1(t)
        a = self.att(h, h, h, need_weights=False)[0]
        t = t + a
        return t + self.ff(self.n2(t))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1))
        self.pos = nn.Parameter(torch.randn(1, D, spec.Y, spec.X) * 0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS) for _ in range(2)])
        self.norm = nn.LayerNorm(D)
        self.h1 = nn.Conv2d(D, HEAD, 3, padding=1)
        self.h2 = nn.Conv2d(HEAD, spec.n_types, 1)
        nn.init.normal_(self.h2.weight, std=0.01)
        nn.init.zeros_(self.h2.bias)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y = obs.shape[2]
        X = obs.shape[3]
        f = self.stem(obs) + self.pos
        t = f.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        f = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.h2(F.relu(self.h1(f))).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
