import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"ppo_epochs": 2, "n_minibatches": 64, "gamma": 0.97, "ent_coef": 0.005, "lr": 0.0007}

D = 64
HEADS = 4
FF = 128
LAYERS = 2
G = 64


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, FF), nn.ReLU(), nn.Linear(FF, D))

    def forward(self, x):
        y = self.n1(x)
        a = self.att(y, y, y, need_weights=False)[0]
        x = x + a
        return x + self.ff(self.n2(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.T = int(spec.n_types)
        cells = self.Y * self.X
        self.embed = nn.Conv2d(spec.in_channels, D, 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, cells, D))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Conv2d(D, self.T, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * D, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        e = self.embed(obs)
        t = e.flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        grid = t.transpose(1, 2).reshape(B, D, Y, X)
        place = self.head(grid).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
