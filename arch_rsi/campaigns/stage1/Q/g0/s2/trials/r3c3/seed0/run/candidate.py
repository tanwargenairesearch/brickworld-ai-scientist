import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HEADS = 4


class ChanLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.footprints = [tuple(fp) for fp in spec.footprints]
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), ChanLN(WIDTH), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), ChanLN(WIDTH), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, spec.Y * spec.X, WIDTH) * 0.02)
        self.attn_ln = nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.proj = nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 1), nn.ReLU())
        self.type_heads = nn.ModuleList([nn.Conv2d(3 * WIDTH, 1, 1) for _ in self.footprints])
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        B, W, Y, X = f.shape
        t = f.flatten(2).transpose(1, 2) + self.pos
        a = self.attn_ln(t)
        a = self.attn(a, a, a, need_weights=False)[0]
        t = t + a
        f = t.transpose(1, 2).reshape(B, W, Y, X)
        h = self.proj(f)
        outs = []
        for (dy, dx), head in zip(self.footprints, self.type_heads):
            hp = F.pad(h, (0, dx - 1, 0, dy - 1))
            mx = F.max_pool2d(hp, (dy, dx), stride=1)
            av = F.avg_pool2d(hp, (dy, dx), stride=1)
            outs.append(head(torch.cat([h, mx, av], dim=1)).flatten(1))
        place = torch.cat(outs, dim=1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
