import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 16, "n_minibatches": 64, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

STEM = 96
W = 40
HH = 32
GH = 64


def small_init(m):
    nn.init.orthogonal_(m.weight, gain=0.01)
    nn.init.zeros_(m.bias)


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = CLN(c)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        C = spec.in_channels
        Y, X = int(spec.Y), int(spec.X)
        n_place = int(spec.n_place)
        k = max(1, min(8, 400000 // (Y * X * n_place)))
        self.stem = nn.Sequential(nn.Conv2d(C, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, W, 1))
        self.blocks = nn.ModuleList([Block(W) for _ in range(2)])
        self.norm = CLN(W)
        self.fconv = nn.ModuleList([nn.Conv2d(W, HH, (dy, dx)) for dy, dx in self.fps])
        self.fout = nn.ModuleList([nn.Conv2d(HH, 1, 1) for _ in self.fps])
        for m in self.fout:
            small_init(m)
        self.fproj = nn.Conv2d(W, k, 1)
        self.flat = nn.Linear(k * Y * X, n_place)
        small_init(self.flat)
        self.done_mlp = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU(), nn.Linear(GH, 1))
        self.val_mlp = nn.Sequential(nn.Linear(2 * W, GH), nn.ReLU(), nn.Linear(GH, 1))
        small_init(self.done_mlp[2])

    def forward(self, obs, mask):
        x = self.stem(obs)
        for b in self.blocks:
            x = b(x)
        f = F.relu(self.norm(x))
        res = []
        for (dy, dx), conv, out in zip(self.fps, self.fconv, self.fout):
            q = F.pad(f, (0, dx - 1, 0, dy - 1))
            res.append(out(F.relu(conv(q))))
        place = torch.cat(res, dim=1).flatten(1)
        place = place + self.flat(F.relu(self.fproj(f)).flatten(1))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        logits = torch.cat([place, self.done_mlp(g)], dim=1)
        return logits, self.val_mlp(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
