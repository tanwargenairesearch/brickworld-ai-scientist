import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 256, "rollout_len": 16, "n_minibatches": 16, "ppo_epochs": 1, "ent_coef": 0.01, "gamma": 0.95, "gae_lambda": 0.9}

STEM = 48
W = 40
LW = 64
SKIP = 16
PROJ = 32
HID = 32
GLOB = 64
CAP = 4.0
DSCALE = 0.05
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


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        t = spec.n_types
        self.stem1 = ortho(nn.Conv2d(c + 2, STEM, 1), RG)
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
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), RG)
        hc = W + SKIP
        self.head = PoolHead(hc, spec.footprints, PROJ, HID)
        self.dense = ortho(nn.Conv2d(hc, t, 3, padding=1), 0.01)
        self.anchor = ortho(nn.Conv2d(hc, 1, 1), 0.01)
        self.glob = ortho(nn.Linear(2 * W + 2 * LW + 2 * SKIP, GLOB), RG)
        self.type_term = ortho(nn.Linear(GLOB, t), 0.01)
        self.done_head = ortho(nn.Linear(GLOB + SKIP, 1), 1.0)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        b = obs.shape[0]
        yy = obs.shape[2]
        xx = obs.shape[3]
        s = F.relu(self.stem2(F.relu(self.stem1(add_coords(obs)))))
        h1 = F.relu(s + self.n1(self.c1(s)))
        d = F.relu(self.nd(self.down(h1)))
        d = F.relu(d + self.nl(self.l1(d)))
        u = F.interpolate(self.up(d), size=(yy, xx), mode="nearest")
        h2 = F.relu(h1 + u)
        f = F.relu(h2 + self.n2(self.c2(h2)))
        k = F.relu(self.skip(obs))
        hf = torch.cat([f, k], dim=1)
        km = k.amax((2, 3))
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), d.mean((2, 3)), d.amax((2, 3)), k.mean((2, 3)), km], dim=1)))
        place = self.head(hf) + self.dense(hf) + self.anchor(hf) + self.type_term(g).view(b, -1, 1, 1)
        done = DSCALE * self.done_head(torch.cat([g, km], dim=1))
        raw = torch.cat([place.flatten(1), done], dim=1)
        logits = CAP * torch.tanh(raw / CAP)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
