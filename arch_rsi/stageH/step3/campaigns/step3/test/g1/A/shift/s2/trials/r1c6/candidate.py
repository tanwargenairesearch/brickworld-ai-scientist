import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.001, "n_envs": 128, "rollout_len": 32, "n_minibatches": 16, "ppo_epochs": 3, "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95}

STEM = 64
WIDTH = 40
SKIP = 16
PROJ = 32
HID = 32
LE = 8
LH = 16
GLOB = 64
RG = 1.4142135623730951


def ortho(m, gain):
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b = obs.shape[0]
    y = obs.shape[2]
    x = obs.shape[3]
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def pool3(p, dy, dx):
    if dy * dx == 1:
        return p, p, p
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return mx, mn, av


class PoolHead(nn.Module):
    def __init__(self, c, footprints, proj, hid):
        super().__init__()
        self.fps = [(int(a), int(b)) for a, b in footprints]
        self.proj = ortho(nn.Conv2d(c, proj, 1), 1.0)
        self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * proj, hid, 1), RG) for fp in self.fps])
        self.outs = nn.ModuleList([ortho(nn.Conv2d(hid, 1, 1), 0.01) for fp in self.fps])

    def forward(self, f):
        p = self.proj(f)
        logits = []
        hids = []
        for (dy, dx), mix, out in zip(self.fps, self.mix, self.outs):
            mx, mn, av = pool3(p, dy, dx)
            h = F.relu(mix(torch.cat([mx, mn, av], dim=1)))
            hids.append(h)
            logits.append(out(h))
        return torch.cat(logits, dim=1), hids


class LayerHead(nn.Module):
    def __init__(self, z, footprints, hid, e, lh):
        super().__init__()
        self.z = z
        self.e = e
        self.fps = [(int(a), int(b)) for a, b in footprints]
        idx = []
        for i in range(z):
            idx = idx + [i, z + i, 2 * z + i, 3 * z + i, 4 * z, 4 * z + 1]
        self.idx = idx
        self.enc = ortho(nn.Conv2d(6 * z, e * z, 1, groups=z), RG)
        self.mix = nn.ModuleList([ortho(nn.Conv2d(3 * e * z, lh * z, 1, groups=z), RG) for fp in self.fps])
        self.out = nn.ModuleList([ortho(nn.Conv2d(lh * z, z, 1, groups=z), 0.01) for fp in self.fps])
        self.lw = nn.ModuleList([ortho(nn.Conv2d(hid, z, 1), 0.01) for fp in self.fps])

    def forward(self, obs, hids):
        b = obs.shape[0]
        yy = obs.shape[2]
        xx = obs.shape[3]
        p = F.relu(self.enc(obs[:, self.idx]))
        res = []
        for (dy, dx), mix, out, lw, h in zip(self.fps, self.mix, self.out, self.lw, hids):
            mx, mn, av = pool3(p, dy, dx)
            pooled = torch.stack([mx, mn, av], dim=1)
            pooled = pooled.view(b, 3, self.z, self.e, yy, xx).transpose(1, 2).reshape(b, 3 * self.z * self.e, yy, xx)
            s = out(F.relu(mix(pooled)))
            w = torch.softmax(lw(h), dim=1)
            res.append((w * s).sum(1, keepdim=True))
        return torch.cat(res, dim=1)


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        c = spec.in_channels
        self.stem1 = ortho(nn.Conv2d(c + 2, STEM, 1), RG)
        self.stem2 = ortho(nn.Conv2d(STEM, WIDTH, 1), RG)
        self.c1 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), RG)
        self.n1 = nn.GroupNorm(8, WIDTH)
        self.c2 = ortho(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), RG)
        self.n2 = nn.GroupNorm(8, WIDTH)
        self.skip = ortho(nn.Conv2d(c, SKIP, 1), RG)
        self.head = PoolHead(WIDTH + SKIP, spec.footprints, PROJ, HID)
        self.lhead = LayerHead(spec.Z, spec.footprints, HID, LE, LH)
        self.anchor = ortho(nn.Conv2d(WIDTH + SKIP, 1, 1), 0.01)
        self.glob = ortho(nn.Linear(2 * WIDTH + SKIP, GLOB), RG)
        self.type_term = ortho(nn.Linear(GLOB, spec.n_types), 0.01)
        self.done_head = ortho(nn.Linear(GLOB, 1), 0.01)
        self.value_head = ortho(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        b = obs.shape[0]
        s = F.relu(self.stem2(F.relu(self.stem1(add_coords(obs)))))
        h = F.relu(self.n1(self.c1(s)))
        f = F.relu(h + self.n2(self.c2(h)))
        k = F.relu(self.skip(obs))
        hf = torch.cat([f, k], dim=1)
        g = F.relu(self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3)), k.amax((2, 3))], dim=1)))
        place, hids = self.head(hf)
        place = place + self.lhead(obs, hids) + self.anchor(hf) + self.type_term(g).view(b, -1, 1, 1)
        logits = torch.cat([place.flatten(1), self.done_head(g)], dim=1)
        return logits, self.value_head(g).squeeze(1)


def build_policy(spec):
    return Net(spec)
