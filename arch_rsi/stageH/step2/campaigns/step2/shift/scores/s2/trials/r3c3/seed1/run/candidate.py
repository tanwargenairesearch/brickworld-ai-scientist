import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 256, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1}

D = 64
NH = 4
LAYERS = 2
G = 64


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, NH, batch_first=True)
        self.n2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 2 * D), nn.ReLU(), nn.Linear(2 * D, D))

    def forward(self, x):
        y = self.n1(x)
        a = self.attn(y, y, y, need_weights=False)[0]
        x = x + a
        x = x + self.ff(self.n2(x))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.nt = int(spec.n_types)
        n = int(spec.Y) * int(spec.X)
        self.embed = nn.Linear(spec.in_channels, D)
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.nf = nn.LayerNorm(D)
        self.out = nn.Conv2d(D, self.nt, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * D, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        t = obs.flatten(2).transpose(1, 2)
        h = self.embed(t) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.nf(h)
        grid = h.transpose(1, 2).reshape(B, D, Y, X)
        place = self.out(grid).flatten(1)
        g = self.glob(torch.cat([h.mean(1), h.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
