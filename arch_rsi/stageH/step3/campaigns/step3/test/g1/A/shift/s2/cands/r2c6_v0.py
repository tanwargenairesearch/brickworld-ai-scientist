import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.002, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.3}

STEM = 48
W = 40
LW = 64
SKIP = 16
PROJ = 32
HID = 32
GLOB = 64
DK = 5
RG = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(t):
    b = t.shape[0]
    y = t.shape[2]
    x = t.shape[3]
    ys = torch.linspace(-1.0, 1.0, y, device=t.device, dtype=t.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=t.device, dtype=t.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([t, ys, xs], dim=1)


def fp_pool(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av], dim=1)


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = ortho(nn.Conv2d(c, proj, 1), 1.0)
        self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * proj, hid, 1), RG) for fp in self.fps])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(hid, 1, 1), 0.01) for fp in self.fps])

    def forward(self, f):
        p = self.proj(f)
        res = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            res.append(out(F.relu(mix(fp_pool(p, dy, dx)))))
        return torch.cat(res, dim=1)


class CellSumHead(nn.Module):
    def __init__(self, c, footprints):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        n = len(self.fps)
        self.u = ortho(nn.Conv2d(c, n, 1), 0.01)
        self.v = ortho(nn.Conv2d(c, n, 1), 0.01)

    def forward(self, f):
        u = self.u(f)
        v = self.v(f)
        res = []
        for i, (dy, dx) in enumerate(self.fps):
            ui = u[:, i:i + 1]
            vi = v[:, i:i + 1]
            if dy * dx == 1:
                res.append(ui + vi)
            else:
                qu = F.pad(ui, (0, dx - 1, 0, dy - 1))
                qv = F.pad(vi, (0, dx - 1, 0, dy - 1))
                s = F.avg_pool2d(qu, (dy, dx), stride=1) * float(dy * dx)
                m = -F.max_pool2d(-qv, (dy, dx), stride=1)
                res.append(s + m)
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        t = spec.n_types
        self.t = t
        self.n_place = spec.n_place
        cin = c + t
        self.stem1 = ortho(nn.Conv2d(cin + 2, STEM, 1), RG)
        self.stem2 = ortho(nn.Conv2d(STEM, W, 1), RG)
        self.c1 = ortho(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n1 = nn.GroupNorm(8, W)
        self.down = ortho(nn.Conv2d(W, LW, 3, stride=2, padding=1), RG)
        self.nd = nn.GroupNorm(8, LW)
        self.l1 = ortho(nn.Conv2d(LW, LW, 3, padding=1), RG)
        self.nl = nn.GroupNorm(8, LW)
        self.up = ortho(nn.Conv2d(LW, W, 1), 1.0)
        self.c2 = ortho(nn.Conv2d(W, W, 3, padding=1), RG)
        self.n2 = nn.GroupNorm(8, W)
        self.skip = ortho(nn.Conv2d(cin, SKIP, 1), RG)
        hc = W + SKIP
        self.head = PoolHead(hc, spec.footprints, PROJ, HID)
        self.cells = CellSumHead(hc, spec.footprints)
        self.dense = ortho(nn.Conv2d(hc, t, DK, padding=DK // 2), 0.01)
        self.anchor = ortho(nn.Conv2d(hc, 1, 1), 0.01)
        self.glob = ortho(nn.Linear(2 * W + 2 * LW + 2 * SKIP, GLOB), RG)
        self.type_term = ortho(nn.Linear(GLOB, t), 0.01)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)
        self.lscale = nn.Parameter(torch.zeros(1))

    def forward(self, obs, mask):
        b = obs.shape[0]
        yy = obs.shape[2]
        xx = obs.shape[3]
        mk = mask[:, :self.n_place].to(obs.dtype).view(b, self.t, yy, xx)
        inp = torch.cat([obs, mk], dim=1)
        s = F.relu(self.stem2(F.relu(self.stem1(add_coords(inp)))))
        h1 = F.relu(s + self.n1(self.c1(s)))
        d = F.relu(self.nd(self.down(h1)))
        d = F.relu(d + self.nl(self.l1(d)))
        u = F.interpolate(self.up(d), size=(yy, xx), mode="nearest")
        h2 = F.relu(h1 + u)
        f = F.relu(h2 + self.n2(self.c2(h2)))
        k = F.relu(self.skip(inp))
        hf = torch.cat([f, k], dim=1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), d.mean((2, 3)), d.amax((2, 3)), k.mean((2, 3)), k.amax((2, 3))], dim=1)))
        place = self.head(hf) + self.cells(hf) + self.dense(hf) + self.anchor(hf) + self.type_term(g).view(b, -1, 1, 1)
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1) * torch.exp(self.lscale)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
