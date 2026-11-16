import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 512, "rollout_len": 8, "n_minibatches": 16, "ppo_epochs": 1, "gae_lambda": 0.85, "ent_coef": 0.01, "clip_eps": 0.2}

W = 40
P = 32
H = 24
G = 64


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(4, c)
        self.c = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        T = len(self.fps)
        self.stem = nn.Sequential(
            nn.Conv2d(spec.in_channels + 2, 64, 1), nn.ReLU(),
            nn.Conv2d(64, W, 1))
        self.blocks = nn.ModuleList([Block(W) for _ in range(2)])
        self.norm = nn.GroupNorm(4, W)
        self.proj = nn.Conv2d(W, P, 1)
        self.mix = nn.Conv2d(3 * P, H, 1)
        self.temb = nn.Parameter(torch.zeros(T, H))
        nn.init.normal_(self.temb, std=0.1)
        self.gproj = nn.Linear(G, H)
        self.mix2 = nn.Conv2d(H, H, 1)
        self.out = nn.Conv2d(H, 1, 1)
        self.glob = nn.Sequential(nn.Linear(2 * W, G), nn.ReLU())
        self.done_head = nn.Linear(G, 1)
        self.value_head = nn.Linear(G, 1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.done_head.weight, std=0.01)
        nn.init.constant_(self.done_head.bias, -2.0)

    def forward(self, obs, mask):
        B, _, Y, X = obs.shape
        ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
        xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
        x = self.stem(torch.cat([obs, ys, xs], dim=1))
        for b in self.blocks:
            x = b(x)
        f = F.relu(self.norm(x))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        p = self.proj(f)
        pooled = torch.stack([fp_pool(p, dy, dx) for dy, dx in self.fps], dim=1)
        T = pooled.shape[1]
        h = self.mix(pooled.reshape(B * T, 3 * P, Y, X)).reshape(B, T, H, Y, X)
        h = h + self.temb.view(1, T, H, 1, 1) + self.gproj(g).view(B, 1, H, 1, 1)
        h = F.relu(h).reshape(B * T, H, Y, X)
        h = F.relu(self.mix2(h))
        place = self.out(h).reshape(B, T * Y * X)
        d = self.done_head(g)
        logits = torch.cat([F.log_softmax(place, dim=1) + F.logsigmoid(-d), F.logsigmoid(d)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
