import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 1.5e-3, "n_envs": 256, "rollout_len": 32, "n_minibatches": 32,
          "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
WIDTH = 32
GROUPS = 8
SKIP = 16
LP = 8
LQ = 4
LH = 20
GL = 96
DET = 32
MAX_ITERS = 5


def ortho(module, gain):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def foot_pool(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


def add_coords(obs):
    B, _, Y, X = obs.shape
    ys = torch.linspace(-1.0, 1.0, Y, device=obs.device, dtype=obs.dtype).view(1, 1, Y, 1).expand(B, 1, Y, X)
    xs = torch.linspace(-1.0, 1.0, X, device=obs.device, dtype=obs.dtype).view(1, 1, 1, X).expand(B, 1, Y, X)
    return torch.cat([obs, ys, xs], dim=1)


class IterTrunk(nn.Module):
    def __init__(self, c, groups, iters):
        super().__init__()
        self.iters = iters
        self.inj = nn.Conv2d(c, c, 1)
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.norm = nn.GroupNorm(groups, c)

    def forward(self, e):
        a = self.inj(e)
        x = e
        for k in range(self.iters):
            x = F.relu(x + self.norm(self.conv(x) + a))
        return x


class LandHead(nn.Module):
    def __init__(self, spec, ctrunk, p, q, hid):
        super().__init__()
        self.Z = int(spec.Z)
        self.p = p
        self.q = q
        self.d = p + 3 * q
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        n = len(self.fps)
        self.v1 = nn.Conv2d(4, p, 1)
        self.v2 = nn.Conv2d(p, p, 1)
        self.pe = nn.Parameter(torch.zeros(1, p, self.Z, 1, 1))
        nn.init.normal_(self.pe, std=0.1)
        self.cb = nn.Conv2d(p, q, 1)
        self.ca = nn.Conv2d(p, q, 1)
        self.mix = nn.Conv2d(2 * self.d, hid, 1)
        self.tproj = nn.Conv2d(ctrunk, hid, 1)
        self.mproj = nn.Conv2d(n, hid, 1)
        self.temb = nn.Parameter(torch.zeros(n, hid, 1, 1, 1))
        self.scores = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        self.gates = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in self.fps])
        self.tbias = nn.Parameter(torch.zeros(n))

    def init_out(self):
        ortho(self.scores, 0.01)
        ortho(self.gates, 0.01)

    def forward(self, obs, f, mimg):
        B, _, Y, X = obs.shape
        Z, P, Q, D = self.Z, self.p, self.q, self.d
        o = obs[:, :4 * Z].reshape(B, 4, Z * Y, X)
        u = F.relu(self.v1(o)).view(B, P, Z, Y, X) + self.pe
        u = F.relu(self.v2(u.reshape(B, P, Z * Y, X)))
        cb = F.relu(self.cb(u)).view(B, Q, Z, Y, X)
        ca = F.relu(self.ca(u)).view(B, Q, Z, Y, X)
        below = (torch.cumsum(cb, dim=2) - cb) / Z
        above = torch.flip(torch.cumsum(torch.flip(ca, [2]), dim=2), [2]) / Z
        prev = F.pad(cb, (0, 0, 0, 0, 1, 0))[:, :, :Z]
        feats = torch.cat([u.view(B, P, Z, Y, X), below, above, prev], dim=1).reshape(B, D * Z, Y, X)
        t = self.tproj(f) + self.mproj(mimg)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            mx, mn, av = foot_pool(feats, dy, dx)
            tp = foot_pool(t, dy, dx)[2]
            m = torch.cat([mx.reshape(B, D, Z * Y, X), mn.reshape(B, D, Z * Y, X)], dim=1)
            h = self.mix(m).view(B, -1, Z, Y, X) + tp.unsqueeze(2) + self.temb[i]
            h = F.relu(h).reshape(B, -1, Z * Y, X)
            s = self.scores[i](h).view(B, Z, Y, X)
            g = self.gates[i](h).view(B, Z, Y, X)
            res.append((torch.softmax(g, dim=1) * s).sum(1, keepdim=True) + self.tbias[i])
        return torch.cat(res, dim=1).flatten(1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.Z = int(spec.Z)
        self.Y = int(spec.Y)
        self.X = int(spec.X)
        self.nt = int(spec.n_types)
        self.npl = int(spec.n_place)
        iters = min(MAX_ITERS, max(3, (max(self.Y, self.X) + 1) // 2))
        self.stem = nn.Sequential(
            nn.Conv2d(c + 2, STEM, 1), nn.ReLU(),
            nn.Conv2d(STEM, WIDTH, 1), nn.GroupNorm(GROUPS, WIDTH), nn.ReLU())
        self.trunk = IterTrunk(WIDTH, GROUPS, iters)
        self.skip = nn.Conv2d(c, SKIP, 1)
        self.head = LandHead(spec, WIDTH + SKIP, LP, LQ, LH)
        self.det = nn.Conv2d(2 * self.Z + 1, DET, 1)
        self.glob = nn.Sequential(nn.Linear(2 * WIDTH + 2 * SKIP + 2 * DET, GL), nn.ReLU())
        self.done_head = nn.Linear(GL, 1)
        self.value_head = nn.Sequential(nn.Linear(GL, GL), nn.ReLU(), nn.Linear(GL, 1))
        ortho(self, math.sqrt(2.0))
        self.head.init_out()
        ortho(self.done_head, 0.01)
        ortho(self.value_head[2], 1.0)

    def forward(self, obs, mask):
        Z = self.Z
        B = obs.shape[0]
        mimg = mask[:, :self.npl].to(obs.dtype).reshape(B, self.nt, self.Y, self.X)
        f = self.trunk(self.stem(add_coords(obs)))
        raw = F.relu(self.skip(obs))
        fr = torch.cat([f, raw], dim=1)
        d = F.relu(self.det(torch.cat([obs[:, 2 * Z:4 * Z], obs[:, 4 * Z + 2:4 * Z + 3]], dim=1)))
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)),
                                 raw.mean((2, 3)), raw.amax((2, 3)),
                                 d.mean((2, 3)), d.amax((2, 3))], dim=1))
        place = self.head(obs, fr, mimg)
        logits = torch.cat([place, self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
