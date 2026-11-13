import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HEADS = 4


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.pos = nn.Parameter(torch.randn(1, spec.Y * spec.X, WIDTH) * 0.02)
        self.ln1 = nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.ReLU(), nn.Linear(2 * WIDTH, WIDTH))
        self.place_head = nn.Conv2d(WIDTH, spec.n_types, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        f = self.body(obs)
        B, W, Y, X = f.shape
        t = f.flatten(2).transpose(1, 2) + self.pos
        h = self.ln1(t)
        a, _ = self.attn(h, h, h, need_weights=False)
        t = t + a
        t = t + self.mlp(self.ln2(t))
        f2 = t.transpose(1, 2).reshape(B, W, Y, X)
        place = self.place_head(f2).flatten(1)
        g = self.glob(torch.cat([f2.mean((2, 3)), f2.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
