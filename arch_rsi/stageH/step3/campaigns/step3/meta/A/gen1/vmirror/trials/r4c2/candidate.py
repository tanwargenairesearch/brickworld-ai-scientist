import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0007, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 2, "ent_coef": 0.01, "gamma": 0.97, "gae_lambda": 0.92, "clip_eps": 0.2}

COL1 = 64
COL2 = 48
UW = 48
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


def add_coords(obs):
    b = obs.shape[0]
    yy = obs.shape[2]
    xx = obs.shape[3]
    cy = torch.linspace(-1.0, 1.0, yy, device=obs.device, dtype=obs.dtype).view(1, 1, yy, 1).expand(b, 1, yy, xx)
    cx = torch.linspace(-1.0, 1.0, xx, device=obs.device, dtype=obs.dtype).view(1, 1, 1, xx).expand(b, 1, yy, xx)
    return torch.cat([obs, cy, cx], dim=1)


class CLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class UCtx(nn.Module):
    def __init__(self, c, w):
        super().__init__()
        self.d1 = nn.Conv2d(c, w, 3, stride=2, padding=1)
        self.g1 = nn.GroupNorm(4, w)
        self.d2 = nn.Conv2d(w, w, 3, stride=2, padding=1)
        self.g2 = nn.GroupNorm(4, w)
        self.m = nn.Conv2d(w, w, 3, padding=1)
        self.gm = nn.GroupNorm(4, w)
        self.gl = nn.Linear(w, w)
        self.u1 = nn.Conv2d(2 * w, w, 3, padding=1)
        self.g3 = nn.GroupNorm(4, w)
        self.u0 = nn.Conv2d(c + w, w, 3, padding=1)
        self.g4 = nn.GroupNorm(4, w)

    def forward(self, x):
        a = F.relu(self.g1(self.d1(x)))
        b2 = F.relu(self.g2(self.d2(a)))
        b2 = F.relu(b2 + self.gm(self.m(b2)) + self.gl(b2.mean((2, 3)))[:, :, None, None])
        up = F.interpolate(b2, size=(a.shape[2], a.shape[3]), mode="nearest")
        a2 = F.relu(self.g3(self.u1(torch.cat([a, up], dim=1))))
        up0 = F.interpolate(a2, size=(x.shape[2], x.shape[3]), mode="nearest")
        return F.relu(self.g4(self.u0(torch.cat([x, up0], dim=1))))


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
        c = spec.in_channels + 2
        self.a1 = nn.Conv2d(c, COL1, 1)
        self.l1 = CLN(COL1)
        self.a2 = nn.Conv2d(COL1, COL2, 1)
        self.l2 = CLN(COL2)
        self.uctx = UCtx(COL2, UW)
        self.head = GatherHead(COL2 + UW, spec.footprints, P, GH)
        self.logt = nn.Parameter(torch.zeros(1))
        self.glob = nn.Sequential(nn.Linear(2 * (COL2 + UW), 64), nn.ReLU())
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
        o = add_coords(obs)
        h = F.relu(self.l1(self.a1(o)))
        h = F.relu(self.l2(self.a2(h)))
        x = self.uctx(h)
        f = torch.cat([h, x], dim=1)
        place = (self.head(f) * torch.exp(self.logt)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        m = F.relu(self.gcell(o))
        q = self.gate(torch.cat([m.amax((2, 3)), m.mean((2, 3)), g], dim=1))
        done = self.gate_out(q) + torch.logsumexp(place, dim=1, keepdim=True).detach()
        logits = torch.cat([place, done], dim=1)
        value = self.value_head(torch.cat([g, q], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
