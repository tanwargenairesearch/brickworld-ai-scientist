import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 64
HEADS = 4
FF = 64


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, WIDTH, 3, padding=1), nn.ReLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU())
        self.attn_norm = nn.LayerNorm(WIDTH)
        self.attn = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.ff_norm = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(nn.Linear(WIDTH, FF), nn.ReLU(), nn.Linear(FF, WIDTH))
        self.out_norm = nn.LayerNorm(WIDTH)
        self.place_head = nn.Linear(WIDTH, spec.n_types)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.ReLU())
        self.done_head = nn.Linear(WIDTH, 1)
        self.value_head = nn.Linear(WIDTH, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        f = self.body(torch.cat([obs, ys, xs], dim=1))
        t = f.flatten(2).transpose(1, 2)
        h = self.attn_norm(t)
        a = self.attn(h, h, h, need_weights=False)[0]
        t = t + a
        t = t + self.ff(self.ff_norm(t))
        t = F.relu(self.out_norm(t))
        place = self.place_head(t).transpose(1, 2).reshape(B, -1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
