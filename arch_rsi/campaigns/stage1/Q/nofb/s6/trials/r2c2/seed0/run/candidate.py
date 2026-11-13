import math
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HEADS = 4
HID = 32


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [(int(fp[0]), int(fp[1])) for fp in spec.footprints]
        n = int(spec.Y) * int(spec.X)
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, n, WIDTH) * 0.02)
        self.ln1 = nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.proj = nn.Linear(WIDTH, WIDTH)
        self.ln2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.ReLU(), nn.Linear(2 * WIDTH, WIDTH))
        self.cell = nn.Conv2d(WIDTH, HID, 1)
        self.type_heads = nn.ModuleList([nn.Conv2d(WIDTH + 2 * HID, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)
        for h in list(self.type_heads) + [self.done_head]:
            nn.init.orthogonal_(h.weight, gain=0.01)
            nn.init.zeros_(h.bias)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        f = self.body(obs)
        N = Y * X
        D = WIDTH // HEADS
        t = f.flatten(2).transpose(1, 2) + self.pos
        h = self.ln1(t)
        qkv = self.qkv(h).reshape(B, N, 3, HEADS, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D), dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(B, N, WIDTH)
        t = t + self.proj(o)
        t = t + self.mlp(self.ln2(t))
        f = t.transpose(1, 2).reshape(B, WIDTH, Y, X)
        c = self.cell(f)
        outs = []
        for ti, (dy, dx) in enumerate(self.footprints):
            cp = F.pad(c, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(cp, (dy, dx), stride=1)
            av = F.avg_pool2d(cp, (dy, dx), stride=1)
            outs.append(self.type_heads[ti](torch.cat([f, mx, av], dim=1)))
        place = torch.cat(outs, dim=1).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
