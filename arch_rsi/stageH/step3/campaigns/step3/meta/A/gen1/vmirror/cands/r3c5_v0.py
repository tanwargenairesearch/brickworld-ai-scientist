import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.005, "gamma": 0.9, "gae_lambda": 0.8, "clip_eps": 0.2}

COLW = 96
CTX = 32
LHID = 64
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


class SoftAndLattice(nn.Module):
    def __init__(self, c, z, footprints):
        super().__init__()
        self.z = z
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.hid = nn.Conv2d(c, LHID, 1)
        self.good = nn.Conv2d(LHID, z, 1)
        self.land = nn.Conv2d(LHID, z, 1)
        self.scale = nn.Parameter(torch.full((len(self.fps),), 0.1))
        self.tb = nn.Parameter(torch.zeros(len(self.fps)))

    def forward(self, f):
        u = F.relu(self.hid(f))
        lg = F.logsigmoid(self.good(u))
        le = self.land(u).clamp(-20.0, 20.0)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            n = dy * dx
            if n == 1:
                a = lg
                o = le
            else:
                qa = F.pad(lg, (0, dx - 1, 0, dy - 1))
                a = F.avg_pool2d(qa, (dy, dx), stride=1) * n
                qe = F.pad(le, (0, dx - 1, 0, dy - 1), value=-20.0)
                o = torch.log(F.avg_pool2d(torch.exp(qe), (dy, dx), stride=1) * n + 1e-12)
            s = torch.logsumexp(a + o, dim=1, keepdim=True)
            res.append(s * self.scale[i] + self.tb[i])
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels + 2
        self.a1 = nn.Conv2d(c, COLW, 1)
        self.l1 = CLN(COLW)
        self.a2 = nn.Conv2d(COLW, COLW, 1)
        self.l2 = CLN(COLW)
        self.c1 = nn.Conv2d(COLW, CTX, 3, padding=1)
        self.n1 = nn.GroupNorm(4, CTX)
        self.c2 = nn.Conv2d(CTX, CTX, 3, padding=1)
        self.n2 = nn.GroupNorm(4, CTX)
        self.lat = SoftAndLattice(COLW + CTX, spec.Z, spec.footprints)
        self.dense = GatherHead(COLW + CTX, spec.footprints, P, GH)
        self.glob = nn.Sequential(nn.Linear(2 * (COLW + CTX), 64), nn.ReLU())
        self.gcell = nn.Conv2d(c, GC, 1)
        self.gate = nn.Sequential(nn.Linear(2 * GC + 64, 64), nn.ReLU())
        self.gate_out = nn.Linear(64, 1)
        self.value_head = nn.Linear(128, 1)
        ortho(self, math.sqrt(2.0))
        ortho(self.lat.good, 1.0)
        ortho(self.lat.land, 1.0)
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
        place = (self.lat(f) + self.dense(f)).flatten(1)
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1))
        m = F.relu(self.gcell(o))
        q = self.gate(torch.cat([m.amax((2, 3)), m.mean((2, 3)), g], dim=1))
        done = self.gate_out(q) + torch.logsumexp(place, dim=1, keepdim=True).detach()
        logits = torch.cat([place, done], dim=1)
        value = self.value_head(torch.cat([g, q], dim=1)).squeeze(1)
        return logits, value


def build_policy(spec):
    return Net(spec)
