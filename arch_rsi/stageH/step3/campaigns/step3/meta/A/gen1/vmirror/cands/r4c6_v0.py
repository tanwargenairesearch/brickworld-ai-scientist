import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 512, "rollout_len": 16, "n_minibatches": 32, "ppo_epochs": 1, "ent_coef": 0.005, "gamma": 0.95, "gae_lambda": 0.9, "clip_eps": 0.2}

STEM = 64
COL = 48
WIDTH = 40
SKIP = 16
P = 32
GH = 48
GC = 32
DONE_BIAS = -2.0


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class GatherHead(nn.Module):
    def __init__(self, c, footprints, p, gh):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = nn.Conv2d(c, p, 1)
        self.mix = nn.ModuleList([nn.Conv2d(p * dy * dx, gh, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(gh, 1, 1) for fp in self.fps])

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
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels + spec.n_types
        self.n_types = spec.n_types
        self.n_place = spec.n_place
        self.stem = nn.Sequential(nn.Conv2d(c, STEM, 1), nn.ReLU(), nn.Conv2d(STEM, COL, 1), nn.ReLU())
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.c1 = nn.Conv2d(COL, WIDTH, 3, padding=1)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.head = GatherHead(WIDTH + SKIP + COL, spec.footprints, P, GH)
        self.logt = nn.Parameter(torch.zeros(1))
        self.glob = nn.Sequential(nn.Linear(2 * (WIDTH + SKIP), 64), nn.ReLU())
        self.gcell = nn.Conv2d(c, GC, 1)
        self.gate = nn.Sequential(nn.Linear(2 * GC + 64, 64), nn.ReLU())
        self.gate_out = nn.Linear(64, 1)
        self.value_head = nn.Linear(128, 1)
        ortho(self, math.sqrt(2.0))
        for o in self.head.outs:
            ortho(o, 0.01)
        ortho(self.gate_out, 0.01)
        nn.init.constant_(self.gate_out.bias, DONE_BIAS)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        b = obs.shape[0]
        yy = obs.shape[2]
        xx = obs.shape[3]
        mp = mask[:, :self.n_place].reshape(b, self.n_types, yy, xx).to(obs.dtype)
        i = torch.cat([obs, mp], dim=1)
        s = self.stem(i)
        k = F.relu(self.skip(i))
        x = F.relu(self.n1(self.c1(s)))
        f = F.relu(x + self.n2(self.c2(x)))
        h = torch.cat([f, k], dim=1)
        place = (self.head(torch.cat([h, s], dim=1)) * torch.exp(self.logt)).flatten(1)
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1))
        m = F.relu(self.gcell(i))
        q = self.gate(torch.cat([m.amax((2, 3)), m.mean((2, 3)), g], dim=1))
        done = self.gate_out(q) + torch.logsumexp(place, dim=1, keepdim=True).detach()
        logits = torch.cat([place, done], dim=1)
        value = self.value_head(torch.cat([g, q], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
