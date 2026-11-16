import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 512, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.3}

COLW = 128
CF = 64
CTX = 32
P = 32
GH = 48


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GatherHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, P, 1)
        self.mix = nn.ModuleList([nn.Conv2d(P * dy * dx, GH, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(GH, 1, 1) for fp in self.fps])

    def forward(self, f):
        p = self.proj(f)
        yy = p.shape[2]
        xx = p.shape[3]
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            q = F.pad(p, (0, dx - 1, 0, dy - 1))
            parts = [q[:, :, i:i + yy, j:j + xx] for i in range(dy) for j in range(dx)]
            h = F.relu(mix(torch.cat(parts, dim=1)))
            res.append(out(h))
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.a1 = nn.Conv2d(c, COLW, 1)
        self.l1 = CLN(COLW)
        self.a2 = nn.Conv2d(COLW, CF, 1)
        self.l2 = CLN(CF)
        self.c1 = nn.Conv2d(CF, CTX, 3, padding=1)
        self.n1 = nn.GroupNorm(4, CTX)
        self.c2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.n2 = nn.GroupNorm(4, CTX)
        self.head = GatherHead(CF + CTX, spec.footprints)
        self.glob = nn.Sequential(nn.Linear(2 * (CF + CTX), 64), nn.ReLU())
        self.done_head = nn.Linear(64, 1)
        self.value_head = nn.Linear(64, 1)
        ortho(self, math.sqrt(2.0))
        for o in self.head.outs:
            ortho(o, 0.01)
        ortho(self.done_head, 0.01)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        h = F.relu(self.l1(self.a1(obs)))
        h = F.relu(self.l2(self.a2(h)))
        x = F.relu(self.n1(self.c1(h)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([h, x], dim=1)
        place = self.head(f)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
