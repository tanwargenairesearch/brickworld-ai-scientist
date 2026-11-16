import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32, "ppo_epochs": 2, "ent_coef": 0.005, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

COL1 = 64
COL2 = 48
CTX = 32
K = 16
LH = 32
NSL = 11
P = 24
GH = 32
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


class SliceLattice(nn.Module):
    def __init__(self, z, cctx, footprints, k, lh):
        super().__init__()
        self.z = z
        self.k = k
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.l1 = nn.Conv2d(NSL, k, 1)
        self.ctx = nn.Conv2d(cctx, z * k, 1)
        self.l2 = nn.Conv2d(k, k, 1)
        self.lc = nn.Conv2d(k, k, 3, padding=1)
        self.lg = nn.GroupNorm(4, k)
        self.mix = nn.ModuleList([nn.Conv2d(k * dy * dx, lh, 1) for dy, dx in self.fps])
        self.outs = nn.ModuleList([nn.Conv2d(lh, 1, 1) for fp in self.fps])

    def slices(self, obs):
        b = obs.shape[0]
        yy = obs.shape[2]
        xx = obs.shape[3]
        z = self.z
        bu = obs[:, 0:z]
        tg = obs[:, z:2 * z]
        ms = obs[:, 2 * z:3 * z]
        ex = obs[:, 3 * z:4 * z]
        bb = F.pad(bu, (0, 0, 0, 0, 1, 0), value=1.0)[:, 0:z]
        ba = F.pad(bu, (0, 0, 0, 0, 0, 1))[:, 1:z + 1]
        mb = F.pad(ms, (0, 0, 0, 0, 1, 0))[:, 0:z]
        hb = obs[:, 4 * z:4 * z + 1].expand(b, z, yy, xx)
        ht = obs[:, 4 * z + 1:4 * z + 2].expand(b, z, yy, xx)
        st = obs[:, 4 * z + 2:4 * z + 3].expand(b, z, yy, xx)
        zp = torch.linspace(0.0, 1.0, z, device=obs.device, dtype=obs.dtype).view(1, z, 1, 1).expand(b, z, yy, xx)
        u = torch.stack([bb, bu, ba, tg, ms, mb, ex, hb, ht, st, zp], dim=2)
        return u.reshape(b * z, NSL, yy, xx)

    def forward(self, obs, ctx):
        b = obs.shape[0]
        yy = obs.shape[2]
        xx = obs.shape[3]
        z = self.z
        k = self.k
        u = self.slices(obs)
        cz = self.ctx(ctx).reshape(b * z, k, yy, xx)
        e = F.relu(self.l1(u) + cz)
        e = F.relu(e + self.l2(e))
        e = F.relu(e + self.lg(self.lc(e)))
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            q = F.pad(e, (0, dx - 1, 0, dy - 1))
            parts = [q[:, :, i:i + yy, j:j + xx] for i in range(dy) for j in range(dx)]
            s = out(F.relu(mix(torch.cat(parts, dim=1)))).reshape(b, z, yy, xx)
            res.append(torch.logsumexp(s, dim=1, keepdim=True))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels + 2
        self.a1 = nn.Conv2d(c, COL1, 1)
        self.l1 = CLN(COL1)
        self.a2 = nn.Conv2d(COL1, COL2, 1)
        self.l2 = CLN(COL2)
        self.c1 = nn.Conv2d(COL2, CTX, 3, padding=1)
        self.n1 = nn.GroupNorm(4, CTX)
        self.c2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.n2 = nn.GroupNorm(4, CTX)
        self.lattice = SliceLattice(spec.Z, COL2 + CTX, spec.footprints, K, LH)
        self.dense = GatherHead(COL2 + CTX, spec.footprints, P, GH)
        self.logt = nn.Parameter(torch.zeros(1))
        self.glob = nn.Sequential(nn.Linear(2 * (COL2 + CTX), 64), nn.ReLU())
        self.gcell = nn.Conv2d(c, GC, 1)
        self.gate = nn.Sequential(nn.Linear(2 * GC + 64, 64), nn.ReLU())
        self.gate_out = nn.Linear(64, 1)
        self.value_head = nn.Linear(128, 1)
        ortho(self, math.sqrt(2.0))
        for o in self.lattice.outs:
            ortho(o, 0.01)
        for o in self.dense.outs:
            ortho(o, 0.01)
        ortho(self.gate_out, 0.01)
        nn.init.constant_(self.gate_out.bias, DONE_BIAS)
        ortho(self.value_head, 1.0)

    def forward(self, obs, mask):
        o = add_coords(obs)
        h = F.relu(self.l1(self.a1(o)))
        h = F.relu(self.l2(self.a2(h)))
        x = F.relu(self.n1(self.c1(h)))
        x = F.relu(x + self.n2(self.c2(x)))
        f = torch.cat([h, x], dim=1)
        place = ((self.lattice(obs, f) + self.dense(f)) * torch.exp(self.logt)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        m = F.relu(self.gcell(o))
        q = self.gate(torch.cat([m.amax((2, 3)), m.mean((2, 3)), g], dim=1))
        done = self.gate_out(q) + torch.logsumexp(place, dim=1, keepdim=True).detach()
        logits = torch.cat([place, done], dim=1)
        value = self.value_head(torch.cat([g, q], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
