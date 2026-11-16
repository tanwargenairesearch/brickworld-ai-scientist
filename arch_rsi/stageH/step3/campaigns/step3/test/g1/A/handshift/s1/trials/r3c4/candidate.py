import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RECIPE = {"lr": 0.0015, "n_envs": 64, "rollout_len": 32, "n_minibatches": 8, "ppo_epochs": 2, "ent_coef": 0.003, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2}

K = 12
W = 40
TOK = 64
PROJ = 24
RAW = 12
HID = 32
GLOB = 64


def init_layer(m, gain=math.sqrt(2.0)):
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def zero_layer(m):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def add_coords(obs):
    b, c, y, x = obs.shape
    ys = torch.linspace(-1.0, 1.0, y, device=obs.device, dtype=obs.dtype).view(1, 1, y, 1).expand(b, 1, y, x)
    xs = torch.linspace(-1.0, 1.0, x, device=obs.device, dtype=obs.dtype).view(1, 1, 1, x).expand(b, 1, y, x)
    return torch.cat([obs, ys, xs], dim=1)


def fp_spread(p, dy, dx):
    if dy * dx == 1:
        return torch.cat([p, p, p, torch.zeros_like(p)], dim=1)
    q = F.pad(p, (0, dx - 1, 0, dy - 1))
    mx = F.max_pool2d(q, (dy, dx), stride=1)
    mn = -F.max_pool2d(-q, (dy, dx), stride=1)
    av = F.avg_pool2d(q, (dy, dx), stride=1)
    return torch.cat([mx, mn, av, mx - mn], dim=1)


def done_stats(maps, fps):
    vals = []
    for m, (dy, dx) in zip(maps, fps):
        y, x = m.shape[2], m.shape[3]
        vals.append(m[:, 0, :max(1, y - dy + 1), :max(1, x - dx + 1)].flatten(1))
    v = torch.cat(vals, dim=1).detach()
    lse = torch.logsumexp(v, dim=1) - math.log(v.shape[1])
    return torch.stack([v.amax(1), lse, v.mean(1)], dim=1)


class LayerEnc(nn.Module):
    def __init__(self, z, k):
        super().__init__()
        self.z = z
        self.l1 = init_layer(nn.Conv2d(5, k, 1))
        self.l2 = init_layer(nn.Conv2d(k, k, 1))

    def forward(self, obs):
        b, c, y, x = obs.shape
        z = self.z
        v = obs[:, :4 * z].reshape(b, 4, z, y * x)
        zc = torch.linspace(0.0, 1.0, z, device=obs.device, dtype=obs.dtype).view(1, 1, z, 1).expand(b, 1, z, y * x)
        h = F.relu(self.l1(torch.cat([v, zc], dim=1)))
        h = F.relu(self.l2(h))
        up = torch.cumsum(h, dim=2) / z
        dn = torch.flip(torch.cumsum(torch.flip(h, [2]), dim=2), [2]) / z
        o = torch.cat([h, up, dn], dim=1)
        return o.reshape(b, -1, y, x)


class MixBlock(nn.Module):
    def __init__(self, c, n, tok):
        super().__init__()
        self.n1 = nn.LayerNorm(c)
        self.t1 = init_layer(nn.Linear(n, tok))
        self.t2 = init_layer(nn.Linear(tok, n), 1.0)
        self.n2 = nn.LayerNorm(c)
        self.c1 = init_layer(nn.Linear(c, 2 * c))
        self.c2 = init_layer(nn.Linear(2 * c, c), 1.0)

    def forward(self, t):
        u = self.n1(t).transpose(1, 2)
        u = self.t2(F.gelu(self.t1(u))).transpose(1, 2)
        t = t + u
        t = t + self.c2(F.gelu(self.c1(self.n2(t))))
        return t


class Net(nn.Module):
    def __init__(self, spec):
        super().__init__()
        z = spec.Z
        cin = spec.in_channels + 2
        n = spec.Y * spec.X
        self.fps = [(int(a), int(b)) for a, b in spec.footprints]
        self.enc = LayerEnc(z, K)
        self.stem = nn.Sequential(init_layer(nn.Conv2d(3 * K * z + cin, W, 1)), nn.GroupNorm(8, W), nn.ReLU())
        self.c0 = init_layer(nn.Conv2d(W, W, 3, padding=1))
        self.n0 = nn.GroupNorm(8, W)
        self.block = MixBlock(W, n, TOK)
        self.proj = init_layer(nn.Conv2d(2 * W, PROJ, 1), 1.0)
        self.raw = init_layer(nn.Conv2d(cin, RAW, 1), 1.0)
        cat = PROJ + RAW
        self.mix = nn.ModuleList([init_layer(nn.Conv2d(4 * cat, HID, 1)) for _ in self.fps])
        self.outs = nn.ModuleList([init_layer(nn.Conv2d(HID, 1, 1), 0.01) for _ in self.fps])
        self.anchor = init_layer(nn.Conv2d(cat, 1, 1), 0.01)
        self.tbias = nn.Parameter(torch.zeros(len(self.fps)))
        self.glob = nn.Sequential(init_layer(nn.Linear(4 * W, GLOB)), nn.ReLU())
        self.done_head = init_layer(nn.Linear(GLOB, 1), 0.01)
        self.done_stat = zero_layer(nn.Linear(3, 1))
        self.value_head = init_layer(nn.Linear(GLOB, 1), 1.0)

    def forward(self, obs, mask):
        x = add_coords(obs)
        e = self.enc(obs)
        s = self.stem(torch.cat([e, x], dim=1))
        h = F.relu(s + self.n0(self.c0(s)))
        b, c, yy, xx = h.shape
        t = self.block(h.flatten(2).transpose(1, 2))
        f = t.transpose(1, 2).reshape(b, c, yy, xx)
        p = torch.cat([self.proj(torch.cat([f, s], dim=1)), self.raw(x)], dim=1)
        a = self.anchor(p)
        maps = []
        for i, (dy, dx) in enumerate(self.fps):
            hh = F.relu(self.mix[i](fp_spread(p, dy, dx)))
            maps.append(self.outs[i](hh) + a + self.tbias[i])
        place = torch.cat(maps, dim=1).flatten(1)
        gf = torch.cat([f.mean((2, 3)), f.amax((2, 3)), s.mean((2, 3)), s.amax((2, 3))], dim=1)
        gh = self.glob(gf)
        done = self.done_head(gh) + self.done_stat(done_stats(maps, self.fps))
        logits = torch.cat([place, done], dim=1)
        return logits, self.value_head(gh).squeeze(1)


def build_policy(spec):
    return Net(spec)
