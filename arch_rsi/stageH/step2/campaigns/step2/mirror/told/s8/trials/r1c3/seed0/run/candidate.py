import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
HEADS = 4
LAYERS = 2
FF = 128

RECIPE = {"ent_coef": 0.02}


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.n2 = nn.LayerNorm(D)
        self.f1 = nn.Linear(D, FF)
        self.f2 = nn.Linear(FF, D)

    def forward(self, x):
        B, N, _ = x.shape
        dh = D // HEADS
        qkv = self.qkv(self.n1(x)).view(B, N, 3, HEADS, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh), dim=-1)
        h = torch.matmul(a, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(h)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.Y = spec.Y
        self.X = spec.X
        n = spec.Y * spec.X
        self.embed = nn.Linear(spec.in_channels + 2, D)
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.1)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D)
        self.place = nn.Linear(D, spec.n_types)
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B = obs.shape[0]
        Y, X = self.Y, self.X
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        inp = torch.cat([obs, ys, xs], dim=1)
        t = inp.flatten(2).transpose(1, 2)
        x = self.embed(t) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        place = self.place(x).transpose(1, 2).reshape(B, -1)
        g = self.glob(torch.cat([x.mean(1), x.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
