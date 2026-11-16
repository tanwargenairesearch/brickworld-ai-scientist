import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"n_envs": 128, "rollout_len": 16, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.01}

STEM = 64
D = 48
HEADS = 4
NLAYER = 2
P = 24
HID = 32


class AttnLayer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        B, N, Dm = x.shape
        hd = Dm // self.h
        qkv = self.qkv(self.n1(x)).view(B, N, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, N, Dm)
        x = x + self.o(y)
        return x + self.ff(self.n2(x))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        n = int(spec.Y) * int(spec.X)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, D, 1), nn.ReLU())
        self.conv = nn.Conv2d(D, D, 3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, n, D) * 0.02)
        self.layers = nn.ModuleList([AttnLayer(D, HEADS) for _ in range(NLAYER)])
        self.norm = nn.LayerNorm(D)
        self.proj = nn.Conv2d(D, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(dy * dx * P, HID, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(HID, 1, 1) for _ in self.fps])
        self.glob = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs, mask):
        B, C, Y, X = obs.shape
        s = self.stem(obs)
        f = F.relu(s + self.conv(s))
        t = f.flatten(2).transpose(1, 2) + self.pos
        for layer in self.layers:
            t = layer(t)
        t = self.norm(t)
        fm = t.transpose(1, 2).reshape(B, D, Y, X)
        p = F.relu(self.proj(fm))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            cells = [q[:, :, i:i + Y, j:j + X] for i in range(dy) for j in range(dx)]
            h = F.relu(mix(torch.cat(cells, dim=1)))
            res.append(out(h))
        place = torch.cat(res, dim=1).flatten(1)
        g = self.glob(torch.cat([t.mean(1), t.amax(1)], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
